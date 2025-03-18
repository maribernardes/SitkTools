import warnings
from typing import TYPE_CHECKING, Any, Mapping, cast, Tuple, Dict

import SimpleITK as sitk
import numpy as np
import torch
from torch.utils.data._utils.collate import np_str_obj_array_pattern
from monai.data import MetaTensor, ImageReader
from monai.data.utils import orientation_ras_lps, is_no_channel
from monai.data.utils import to_affine_nd, affine_to_spacing
from monai.data.image_writer import ImageWriter
from monai.config import DtypeLike, KeysCollection, NdarrayOrTensor
from monai.utils import convert_to_dst_type, ensure_tuple, ensure_tuple_rep, MetaKeys, SpaceKeys, TraceKeys
from monai.utils import get_equivalent_dtype, convert_data_type, GridSampleMode, GridSamplePadMode
from monai.transforms import Transform, MapTransform, EnsureChannelFirst
from monai.utils.enums import PostFix

DEFAULT_POST_FIX = PostFix.meta()

# Implements a loader that takes a list of sitk images and outputs as metatensors with respective metadata
# Based on original ITKReader, by replacing the read function (no need to read from file) and adapting from itk object to sitk object
# ATTENTION: I removed the part that dealt with extra channels since sitk support to multichannel (4D) images is limited. 
# That is why this reader lacks the argument channel_dim. It assumes that channel_dim is always None, 
# which results in metadata 'original_channel_dim' to be either 'no_channel' or -1
class sitkReader(ImageReader):
    def __init__(
            self,
            series_name: str = "",
            reverse_indexing: bool = False,
            series_meta: bool = False,
            affine_lps_to_ras: bool = True,
            **kwargs,
        ):
        super().__init__()
        self.kwargs = kwargs
        self.series_name = series_name
        self.reverse_indexing = reverse_indexing
        self.series_meta = series_meta
        self.affine_lps_to_ras = affine_lps_to_ras
    
    # Does nothing, but is required for ImageReader class
    def read(self, img):
        return img
    # Does nothing, but is required for ImageReader class
    def verify_suffix(self, img) -> bool:
        return True
    
    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        img_array: list[np.ndarray] = []
        compatible_meta: dict = {}
        data = self._get_array_data(img)
        img_array.append(data)
        header = self._get_meta_dict(img)
        header[MetaKeys.ORIGINAL_AFFINE] = self._get_affine(img, self.affine_lps_to_ras)
        header[MetaKeys.SPACE] = SpaceKeys.RAS if self.affine_lps_to_ras else SpaceKeys.LPS
        header[MetaKeys.AFFINE] = header[MetaKeys.ORIGINAL_AFFINE].copy()
        header[MetaKeys.SPATIAL_SHAPE] = self._get_spatial_shape(img)
        # Here I removed the option of having multichannel original images. The code default to "no_channel" or -1
        header[MetaKeys.ORIGINAL_CHANNEL_DIM] = (float("nan") if len(data.shape) == len(header[MetaKeys.SPATIAL_SHAPE]) else -1)
        self._copy_compatible_dict(header, compatible_meta)
        return self._stack_images(img_array, compatible_meta), compatible_meta
        
    def _get_meta_dict(self, img) -> dict:
        img_meta_dict = img.GetMetaDataKeys()
        meta_dict: dict = {}
        for key in img_meta_dict:
            if key.startswith("ITK_"):
                continue
            val = img.GetMetaData(key)
            meta_dict[key] = np.asarray(val) if type(val).__name__.startswith("itk") else val
        meta_dict["spacing"] = np.asarray(img.GetSpacing())
        return dict(meta_dict)

    def _get_affine(self, img, lps_to_ras: bool = True):
        dir_array = img.GetDirection()
        direction = np.array([dir_array[0:3],dir_array[3:6],dir_array[6:9]])
        spacing = np.asarray(img.GetSpacing())
        origin = np.asarray(img.GetOrigin())
        sr = min(max(direction.shape[0], 1), 3)
        affine: np.ndarray = np.eye(sr + 1)
        affine[:sr, :sr] = direction[:sr, :sr] @ np.diag(spacing[:sr])
        affine[:sr, -1] = origin[:sr]
        if lps_to_ras:
            affine = orientation_ras_lps(affine)
        return affine

    def _get_spatial_shape(self, img):
        ## Not handling multichannel images with SimpleITK
        dir_array = img.GetDirection()
        sr = np.array([dir_array[0:3],dir_array[3:6],dir_array[6:9]]).shape[0]
        sr = max(min(sr, 3), 1)
        _size = list(img.GetSize())
        return np.asarray(_size[:sr])

    def _get_array_data(self, img):
        ## Not handling multichannel images with SimpleITK
        np_img = sitk.GetArrayFromImage(img)
        return np_img if self.reverse_indexing else np_img.T
    
    def _stack_images(self, image_list: list, meta_dict: dict):
        if len(image_list) <= 1:
            return image_list[0]
        if not is_no_channel(meta_dict.get(MetaKeys.ORIGINAL_CHANNEL_DIM, None)):
            channel_dim = int(meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM])
            return np.concatenate(image_list, axis=channel_dim)
        # stack at a new first dim as the channel dim, if `'original_channel_dim'` is unspecified
        meta_dict[MetaKeys.ORIGINAL_CHANNEL_DIM] = 0
        return np.stack(image_list, axis=0)

    def _copy_compatible_dict(self, from_dict: dict, to_dict: dict):
        if not isinstance(to_dict, dict):
            raise ValueError(f"to_dict must be a Dict, got {type(to_dict)}.")
        if not to_dict:
            for key in from_dict:
                datum = from_dict[key]
                if isinstance(datum, np.ndarray) and np_str_obj_array_pattern.search(datum.dtype.str) is not None:
                    continue
                to_dict[key] = str(TraceKeys.NONE) if datum is None else datum  # NoneType to string for default_collate
        else:
            affine_key, shape_key = MetaKeys.AFFINE, MetaKeys.SPATIAL_SHAPE
            if affine_key in from_dict and not np.allclose(from_dict[affine_key], to_dict[affine_key]):
                raise RuntimeError(
                    "affine matrix of all images should be the same for channel-wise concatenation. "
                    f"Got {from_dict[affine_key]} and {to_dict[affine_key]}."
                )
            if shape_key in from_dict and not np.allclose(from_dict[shape_key], to_dict[shape_key]):
                raise RuntimeError(
                    "spatial_shape of all images should be the same for channel-wise concatenation. "
                    f"Got {from_dict[shape_key]} and {to_dict[shape_key]}."
            )

# Custom version of LoadImage (from monai.transforms.io.array) 
# Instead of receiving a list of filenames, receives a list of sitk image objects
# The transform returns a MetaTensor (data + metadata)
# ATTENTION: I did not implement the option set_track_meta(False) which should return a torch.Tensor
# ATTENTION2: I did not implement the option for multichannel (4D) sitk images. This requires more work in the implemented reader (sitkReader)                
class LoadSitkImage(Transform):
    def __init__(self,
            image_only: bool = False,
            dtype: DtypeLike or None = np.float32,
            ensure_channel_first: bool = False,
            simple_keys: bool = False,
            prune_meta_pattern: str or None = None,
            prune_meta_sep: str = ".",   
        ) -> None:
        self.reader = sitkReader()
        self.image_only = image_only
        self.ensure_channel_first = ensure_channel_first
        self.dtype = dtype
        self.simple_keys = simple_keys
        self.pattern = prune_meta_pattern
        self.sep = prune_meta_sep

    def __call__(self, img):
        if not isinstance(img, sitk.SimpleITK.Image):
            raise RuntimeError(f"{self.__class__.__name__} The input image is not an SimpleITK object.\n")    
        img_array, meta_data = self.reader.get_data(img)
        img_array = convert_to_dst_type(img_array, dst=img_array, dtype=self.dtype)[0]
        if not isinstance(meta_data, dict):
            raise ValueError(f"`meta_data` must be a dict, got type {type(meta_data)}.")
        # Here I changed from original LoadImage to use a torch.tensor instead of numpy array (img_array) 
        # I did this to get similar results from LoadImage when loading from a nifti file
        img = MetaTensor.ensure_torch_and_prune_meta(
            torch.from_numpy(img_array), meta_data, self.simple_keys, pattern=self.pattern, sep=self.sep
            # img_array, meta_data, self.simple_keys, pattern=self.pattern, sep=self.sep
        )
        if self.ensure_channel_first:
            img = EnsureChannelFirst()(img)
        if self.image_only:
            return img
        return img, img.meta if isinstance(img, MetaTensor) else meta_data

# Dictionary-based wrapper of custom LoadSitkImage
# Basically a copy of LoadImaged with the loader replaced by LoadSitkImage and the reader fixed to be sitkReader 
class LoadSitkImaged(MapTransform):
    def __init__(self,
            keys: KeysCollection,
            dtype: DtypeLike = np.float32,
            meta_keys: KeysCollection or None=None,
            meta_key_postfix: str=DEFAULT_POST_FIX,
            overwriting: bool=False,
            image_only: bool=False,
            ensure_channel_first: bool=False,
            simple_keys: bool=False,
            prune_meta_pattern: str or None=None,
            prune_meta_sep: str=".",
            allow_missing_keys: bool=False,
        ):
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadSitkImage(
            image_only,
            dtype,
            ensure_channel_first,
            simple_keys,
            prune_meta_pattern,
            prune_meta_sep
        ) 
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError(
                f"meta_keys should have the same length as keys, got {len(self.keys)} and {len(self.meta_keys)}."
            )
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting
        
    def __call__(self, img):
        d = dict(img)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            img = self._loader(d[key])
            if self._loader.image_only:
                d[key] = img
            else:
                if not isinstance(img, (tuple, list)):
                    raise ValueError(
                        f"loader must return a tuple or list (because image_only=False was used), got {type(img)}."
                    )
                d[key] = img[0]
                if not isinstance(img[1], dict):
                    raise ValueError(f"metadata must be a dict, got {type(img[1])}.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = img[1]
        return d


# Basically a copy from ITKWriter. Only differences are in:
# create_backend_obj: creates a SimpleITK Object and converts the metadata affine accordingly
# write: now returns a SimpleITK Object
class sitkWriter(ImageWriter):
    def __init__(self,
            output_dtype: DtypeLike or None=np.float32, 
            affine_lps_to_ras: bool or None=True,
            **kwargs
        ):
        super().__init__(output_dtype=output_dtype, affine_lps_to_ras=affine_lps_to_ras, affine=None, channel_dim=0, **kwargs)
        
    def set_data_array(self, data_array: NdarrayOrTensor, channel_dim: int or None=0, squeeze_end_dims: bool=True, **kwargs):
        n_chns = data_array.shape[channel_dim] if channel_dim is not None else 0
        self.data_obj = self.convert_to_channel_last(
            data=data_array,
            channel_dim=channel_dim,
            squeeze_end_dims=squeeze_end_dims,
            spatial_ndim=kwargs.pop("spatial_ndim", 3),
            contiguous=kwargs.pop("contiguous", True),
        )
        self.channel_dim = -1  # in most cases, the data is set to channel last
        if squeeze_end_dims and n_chns <= 1:  # num_channel==1 squeezed
            self.channel_dim = None
        if not squeeze_end_dims and n_chns < 1:  # originally no channel and convert_to_channel_last added a channel
            self.channel_dim = None
            self.data_obj = self.data_obj[..., 0]
    
    def set_metadata(self, meta_dict: Mapping or None = None, resample: bool = True, **options):
        original_affine, affine, spatial_shape = self.get_meta_info(meta_dict)
        if self.output_dtype is None and hasattr(self.data_obj, "dtype"):  # pylint: disable=E0203
            self.output_dtype = self.data_obj.dtype  # type: ignore
        self.data_obj, self.affine = self.resample_if_needed(
            data_array=cast(NdarrayOrTensor, self.data_obj),
            affine=affine,
            target_affine=original_affine if resample else None,
            output_spatial_shape=spatial_shape if resample else None,
            mode=options.pop("mode", GridSampleMode.BILINEAR),
            padding_mode=options.pop("padding_mode", GridSamplePadMode.BORDER),
            align_corners=options.pop("align_corners", False),
            dtype=options.pop("dtype", np.float64),
        )
    
    def create_backend_obj(
            self, # Not sure if needed
            data_array: NdarrayOrTensor,
            channel_dim: int or None=0,
            affine: NdarrayOrTensor or None=None,
            dtype: DtypeLike = np.float32,
            affine_lps_to_ras: bool or None=True,
            **kwargs,
        )-> sitk.SimpleITK.Image:
        if isinstance(data_array, MetaTensor) and affine_lps_to_ras is None:
            # do the converting from LPS to RAS only if the space type is currently LPS.
            affine_lps_to_ras = (data_array.meta.get(MetaKeys.SPACE, SpaceKeys.LPS) != SpaceKeys.LPS)  
        data_array = convert_data_type(data_array, np.ndarray)[0]
        _is_vec = channel_dim is not None
        if _is_vec:
            data_array = np.moveaxis(data_array, -1, 0)  # from channel last to channel first
        data_array = data_array.T.astype(get_equivalent_dtype(dtype, np.ndarray), copy=True, order="C")
        sitk_image = sitk.GetImageFromArray(data_array, isVector=_is_vec)
        d = len(sitk_image.GetSize())
        if affine is None:
            affine = np.eye(d + 1, dtype=np.float64)
        _affine = convert_data_type(affine, np.ndarray)[0]
        if affine_lps_to_ras:
            _affine = orientation_ras_lps(to_affine_nd(d, _affine))
        spacing = affine_to_spacing(_affine, r=d)
        _direction: np.ndarray = np.diag(1 / spacing)
        _direction = _affine[:d, :d] @ _direction
        sitk_image.SetSpacing(spacing.tolist())
        sitk_image.SetOrigin(_affine[:d, -1].tolist())
        sitk_image.SetDirection(_direction.ravel().tolist())
        return sitk_image
    
    # Maybe not necessary
    def write(self, verbose: bool = False, **kwargs):
        super().write('sitkImage', verbose=verbose)
        self.data_obj = self.create_backend_obj(
            cast(NdarrayOrTensor, self.data_obj),
            channel_dim=self.channel_dim,
            affine=self.affine,
            dtype=self.output_dtype,
            affine_lps_to_ras=self.affine_lps_to_ras,  # type: ignore
            **kwargs,
        )
        return self.data_obj    

# Push the image (in the form of torch tensor or numpy ndarray) and metadata dictionary into sitk image object.
# Kept most of the parameters from LoadImage, removing only the ones related to file saving
class PushSitkImage(Transform):
    def __init__(self,
            output_dtype: DtypeLike or None = np.float32,
            dtype: DtypeLike or None=np.float32, 
            resample: bool = True,
            mode: str = "nearest",
            padding_mode: str = GridSamplePadMode.BORDER,
            scale: int or None = None,
            squeeze_end_dims: bool = True,
            print_log: bool = True,
            channel_dim: int or None = 0,
        ) -> sitk.SimpleITK.Image:
        self.init_kwargs = {"output_dtype": output_dtype,"scale": scale}
        self.data_kwargs = {"squeeze_end_dims": squeeze_end_dims, "channel_dim": channel_dim}
        self.meta_kwargs = {"resample": resample, "mode": mode, "padding_mode": padding_mode, "dtype": dtype}
        self.write_kwargs = {"verbose": print_log}
        self.writer = sitkWriter(**self.init_kwargs)
        
    # Probably not needed
    def set_options(self, init_kwargs=None, data_kwargs=None, meta_kwargs=None, write_kwargs=None):
        if init_kwargs is not None:
            self.init_kwargs.update(init_kwargs)
        if data_kwargs is not None:
            self.data_kwargs.update(data_kwargs)
        if meta_kwargs is not None:
            self.meta_kwargs.update(meta_kwargs)
        if write_kwargs is not None:
            self.write_kwargs.update(write_kwargs)
        return self
        
    def __call__(self, img: torch.Tensor or np.ndarray, meta_data: dict or None = None):
        meta_data = img.meta if isinstance(img, MetaTensor) else meta_data
        if meta_data:
            meta_spatial_shape = ensure_tuple(meta_data.get("spatial_shape", ()))
            if len(meta_spatial_shape) >= len(img.shape):
                self.data_kwargs["channel_dim"] = None
            elif is_no_channel(self.data_kwargs.get("channel_dim")):
                warnings.warn(
                    f"data shape {img.shape} (with spatial shape {meta_spatial_shape}) "
                    f"but SaveImage `channel_dim` is set to {self.data_kwargs.get('channel_dim')} no channel."
                )
        self.writer.set_data_array(data_array=img, **self.data_kwargs)
        self.writer.set_metadata(meta_dict=meta_data, **self.meta_kwargs)
        return self.writer.write(**self.write_kwargs)

# Dictionary-based wrapper of custom PushSitkImage
# Basically a copy of SaveImaged with the saver replaced by PushSitkImage and the output being a dict of sitk images
class PushSitkImaged(MapTransform):
    def __init__(self,
        keys: KeysCollection,
        meta_keys: KeysCollection or None = None,
        meta_key_postfix: str = DEFAULT_POST_FIX,
        resample: bool = True,
        mode: str = "nearest",
        padding_mode: str = GridSamplePadMode.BORDER,
        scale: int or None = None,
        dtype: DtypeLike = np.float64,
        output_dtype: DtypeLike or None = np.float32,
        allow_missing_keys: bool = False,
        squeeze_end_dims: bool = True,
        print_log: bool = True,
        ):
        super().__init__(keys, allow_missing_keys)
        self.meta_keys = ensure_tuple_rep(meta_keys, len(self.keys))
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.saver = PushSitkImage(
            resample=resample,
            mode=mode,
            padding_mode=padding_mode,
            scale=scale,
            dtype=dtype,
            output_dtype=output_dtype,
            squeeze_end_dims=squeeze_end_dims,
            print_log=print_log,
        )

    # Probably not needed
    def set_options(self, init_kwargs=None, data_kwargs=None, meta_kwargs=None, write_kwargs=None):
        self.saver.set_options(init_kwargs, data_kwargs, meta_kwargs, write_kwargs)
        return self

    def __call__(self, data):
        d = dict(data)
        img = dict()
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            if meta_key is None and meta_key_postfix is not None:
                meta_key = f"{key}_{meta_key_postfix}"
            meta_data = d.get(meta_key) if meta_key is not None else None
            # Adapt to output the sitk image
            img[key] = self.saver(img=d[key], meta_data=meta_data)
        return img
    

class SelectChannel(Transform):
    """
    Select specific channels from a multi-channel tensor.

    Args:
        indices (int or list of int): Indices of the channels to keep.

    Example:
        transform = SelectChannel(indices=0)  # Keep only the first channel
    """

    def __init__(self, indices):
        self.indices = indices if isinstance(indices, (list, tuple)) else [indices]

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.ndim < 3:
            raise ValueError(f"Expected a multi-channel tensor, but got shape {data.shape}")

        return data[self.indices]  # Select specified channels

class SelectChanneld(MapTransform):
    """
    Dictionary-based transform to select specific channels from a multi-channel tensor.

    Args:
        keys (str or list of str): Key(s) of the tensors to modify.
        indices (int or list of int): Indices of the channels to keep.

    Example:
        transform = SelectChanneld(keys="image", indices=0)  # Keep only the first channel
    """

    def __init__(self, keys, indices):
        super().__init__(keys)
        self.transform = SelectChannel(indices)

    def __call__(self, data):
        for key in self.keys:
            data[key] = self.transform(data[key])  # Call the tensor transform
        return data



class ScaleIntensityPerChannel(Transform):
    """
    Scales intensity values per channel independently to the given min/max range.

    Args:
        min_max_values (list of tuples): List of (minv, maxv) for each channel.
    """

    def __init__(self, min_max_values):
        self.min_max_values = min_max_values

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[0] != len(self.min_max_values):
            raise ValueError(f"Expected {len(self.min_max_values)} channels, but got {data.shape[0]}")

        scaled_channels = []
        for i, (minv, maxv) in enumerate(self.min_max_values):
            min_intensity, max_intensity = data[i].min(), data[i].max()
            if max_intensity == min_intensity:  # Avoid division by zero
                scaled = torch.full_like(data[i], minv)
            else:
                scaled = ((data[i] - min_intensity) / (max_intensity - min_intensity)) * (maxv - minv) + minv
            scaled_channels.append(scaled)

        return torch.stack(scaled_channels, dim=0)


class ScaleIntensityPerChanneld(MapTransform):
    """
    Dictionary-based transform to scale intensity values per channel independently.

    Args:
        keys (list): Keys to apply transformation.
        min_max_values (list of tuples): List of (minv, maxv) for each channel.
    """

    def __init__(self, keys, min_max_values):
        super().__init__(keys)
        self.transform = ScaleIntensityPerChannel(min_max_values)

    def __call__(self, data):
        for key in self.keys:
            data[key] = self.transform(data[key])  # Call tensor version
        return data


class MagPhaseToRealImag(Transform):
    """
    MONAI Transform to convert a tensor with magnitude in the first channel and phase in the second channel
    into a tensor with real and imaginary components. Supports both 2D and 3D images.

    Input:
        - Tensor shape: (2, H, W) or (2, D, H, W) where:
            - Channel 0: Magnitude
            - Channel 1: Phase (radians)

    Output:
        - Tensor shape: (2, H, W) or (2, D, H, W) where:
            - Channel 0: Real part
            - Channel 1: Imaginary part
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.ndim not in [3, 4] or data.shape[0] != 2:
            raise ValueError("Input tensor must have shape (2, H, W) for 2D or (2, D, H, W) for 3D.")

        magnitude = data[0]
        phase = data[1]

        # Convert to Real and Imaginary parts
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)

        return torch.stack((real, imag), dim=0)


class MagPhaseToRealImagd(MapTransform):
    """
    Dictionary-based MONAI Transform to convert a concatenated magnitude-phase tensor 
    into real and imaginary components. Supports both 2D and 3D images.

    This assumes the input tensor has been combined using `ConcatItemsd`, with:
        - Channel 0: Magnitude
        - Channel 1: Phase (radians)

    The transform **does not change the key name**, only modifies its contents.

    Args:
        keys (list): List containing a **single key** that holds the (2, H, W) or (2, D, H, W) concatenated tensor.
    """

    def __init__(self, keys):
        super().__init__(keys)
        if len(keys) != 1:
            raise ValueError("keys must contain exactly one element, referring to the concatenated (magnitude, phase) tensor.")

        self.transform = MagPhaseToRealImag()

    def __call__(self, data):
        key = self.keys[0]
        data[key] = self.transform(data[key])  # Call tensor version
        return data


class ComplexToMagPhase(Transform):
    """
    Convert a complex-valued tensor into a 2-channel tensor with magnitude and phase.

    Input: (H, W) or (D, H, W) complex tensor
    Output: (2, H, W) or (2, D, H, W) tensor with:
        - Channel 0: Magnitude
        - Channel 1: Phase
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(data):
            raise ValueError("Input tensor must be a complex-valued tensor.")

        magnitude = torch.abs(data)
        phase = torch.angle(data)
        return torch.stack((magnitude, phase), dim=0)

    
class ComplexToMagPhased(MapTransform):
    """
    Dictionary-based transform to convert a complex-valued tensor into a 2-channel tensor (magnitude, phase).

    Args:
        keys (str): Input key for the complex tensor.
        name (str): Output key for the 2-channel (magnitude, phase) tensor.

    Example:
        transform = ComplexToMagPhased(keys="complex_image", name="mag_phase_tensor")
    """

    def __init__(self, keys, name):
        super().__init__(keys)
        self.input_key = keys
        self.name = name
        self.transform = ComplexToMagPhase()

    def __call__(self, data):
        complex_tensor = data[self.input_key]

        # Convert to magnitude and phase using the tensor transform
        data[self.name] = self.transform(complex_tensor)

        # Remove the intermediate complex representation
        del data[self.input_key]
        return data



class MagPhaseToComplex(Transform):
    """
    Convert a 2-channel tensor (magnitude, phase) into a complex-valued tensor.

    Input: (2, H, W) or (2, D, H, W) tensor
    Output: (H, W) or (D, H, W) complex tensor
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[0] != 2:
            raise ValueError("Input tensor must have two channels: (magnitude, phase).")

        magnitude = data[0]
        phase = data[1]
        return magnitude * torch.exp(1j * phase)


class ComplexToMagPhased(MapTransform):
    """
    Dictionary-based transform to convert a 2-channel tensor (complex-valued) into magnitude and phase.

    Args:
        keys (str): Input key for the 2-channel (real, imag) tensor.
        name (str): Output key for the converted (2, H, W) or (2, D, H, W) tensor.

    Example:
        transform = ComplexToMagPhased(keys="complex_tensor", name="mag_phase_tensor")
    """

    def __init__(self, keys, name):
        super().__init__(keys)
        self.input_key = keys
        self.name = name
        self.transform = ComplexToMagPhase()

    def __call__(self, data):
        tensor = data[self.input_key]

        if tensor.shape[0] != 2:
            raise ValueError("Expected input tensor with shape (2, H, W) or (2, D, H, W), containing real and imaginary channels.")

        # Convert to magnitude and phase using the tensor transform
        data[self.name] = self.transform(torch.view_as_complex(tensor.movedim(0, -1)))  # Convert to complex format

        # Remove the intermediate input key
        del data[self.input_key]
        return data


class RandKSpaceSpikeNoiseMagPhase(Transform):
    """
    Applies randomized k-space spike noise to a 2D or 3D (magnitude, phase) tensor.

    - **Channel 0:** Magnitude
    - **Channel 1:** Phase (in radians)

    Supports input tensors with shape (2, H, W) or (2, D, H, W).
    If the input shape is (2, W, H, D), set `reorder_axes=True` to fix the order before processing.

    Args:
        prob (float): Probability of applying the transform.
        intensity_range (tuple): Min and max random intensity for the k-space spike (in log scale).
        reorder_axes (bool): If `True`, swaps (2, W, H, D) → (2, D, H, W) before processing.
        phase_range (tuple): Optional (min, max) range for the phase values. Defaults to (-π, π).
    """

    def __init__(self, prob=0.5, intensity_range=(11, 12), reorder_axes=False, phase_range=(-torch.pi, torch.pi)):
        self.prob = prob
        self.intensity_range = intensity_range
        self.reorder_axes = reorder_axes
        self.phase_range = phase_range

        if self.phase_range[1] <= self.phase_range[0]:
            raise ValueError(f"Invalid phase range: {self.phase_range}. Expected (min, max) with max > min.")

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        if data.shape[0] != 2:
            raise ValueError("Expected a 2-channel tensor with (Magnitude, Phase).")

        # Apply axis reordering if needed
        if self.reorder_axes and len(data.shape) == 4:  # If 3D (2, W, H, D)
            data = data.permute(0, 3, 2, 1)  # Swap to (2, D, H, W)
            reorder_back = True
        else:
            reorder_back = False

        # Extract magnitude and phase
        magnitude = data[0]
        phase = data[1]

        # Convert to complex representation
        complex_image = magnitude * torch.exp(1j * phase)

        # Apply FFT to move to k-space
        kspace = torch.fft.fftn(complex_image)
        kspace = torch.fft.fftshift(kspace)  # Shift zero frequency to center

        # Determine spike location uniformly across k-space
        spatial_dims = magnitude.shape
        spike_location = tuple(torch.randint(0, dim, (1,)).item() for dim in spatial_dims)

        # Sample spike intensity from a log-uniform distribution
        log_intensity = torch.FloatTensor(1).uniform_(*self.intensity_range).item()
        intensity = 10 ** log_intensity
        spike_value = intensity * (torch.randn(()).item() + 1j * torch.randn(()).item())

        # Apply k-space spike
        kspace[spike_location] += spike_value

        # Enforce Hermitian symmetry
        symmetric_location = tuple((dim - loc) % dim for loc, dim in zip(spike_location, spatial_dims))
        kspace[symmetric_location] += spike_value.conjugate()

        # Apply inverse FFT to return to image domain
        kspace = torch.fft.ifftshift(kspace)  # Reverse shift before IFFT
        noisy_complex_image = torch.fft.ifftn(kspace)

        # Convert back to magnitude and phase
        magnitude_noisy = torch.abs(noisy_complex_image)

        # Ensure phase values are wrapped within the given range
        phase_min, phase_max = self.phase_range
        phase_range_span = phase_max - phase_min
        phase_noisy = ((torch.angle(noisy_complex_image) - phase_min) % phase_range_span) + phase_min

        # Stack back into the original format
        output = torch.stack((magnitude_noisy, phase_noisy), dim=0)

        # If the input was (2, W, H, D), reorder it back
        if reorder_back:
            output = output.permute(0, 3, 2, 1)  # Swap back to (2, W, H, D)

        return output



class RandKSpaceSpikeNoiseMagPhased(MapTransform):
    """
    Dictionary-based transform applying k-space spike noise to a 2D or 3D (magnitude, phase) tensor.

    Ensures k-space symmetry and proper spike scaling for visible striping patterns.

    Args:
        keys (str): Key of the input tensor in the dictionary.
        prob (float): Probability of applying the transform.
        intensity_range (tuple): Min and max random intensity for the k-space spike.
        location_range (tuple, optional): (x_min, x_max, y_min, y_max) range for spike placement.
        reorder_axes (bool): If `True`, swaps (2, W, H, D) → (2, D, H, W) before processing.
    """

    def __init__(self, keys, prob=0.5, intensity_range=(10, 50), reorder_axes=False, phase_range=(-np.pi, np.pi)):
        super().__init__(keys)
        self.prob = prob
        self.intensity_range = intensity_range
        self.reorder_axes = reorder_axes
        self.phase_range = phase_range

    def __call__(self, data):
        key = self.keys[0]
        tensor = data[key]

        if tensor.shape[0] != 2:
            raise ValueError(f"Expected a 2-channel tensor (Magnitude, Phase), but got shape {tensor.shape}")

        transform = RandKSpaceSpikeNoiseMagPhase(
            prob=self.prob,
            intensity_range=self.intensity_range,
            reorder_axes=self.reorder_axes,
            phase_range=self.phase_range
        )

        data[key] = transform(tensor)
        return data
