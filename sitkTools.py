import numpy as np
import SimpleITK as sitk

import math
import itertools

from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow
from copy import copy
from ipywidgets import interact

AX_DIR = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
COR_DIR = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -1.0, 0.0) 
SAG_DIR = (0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
##########################################################
## UTILS
# Get the index of the list element with closest value to provided target
def get_closest_index(target, list):
    return min(enumerate(list), key=lambda x: abs(x[1] - target))[0]

# Read an image from NRRD file based on the image info array
def read_image(iminfo, cols, image_dir, num=0, pixel_id=sitk.sitkFloat32):
    path = image_dir + '/' + iminfo[num, cols.index('filename')]
    image = sitk.ReadImage(path, pixel_id)
    return image

##########################################################
## VISUALIZATION
# Show a pair of images from given arrays
def show_array_pair(array_m, array_p, title=None, patch=None, margin=0.1, dpi=80, cmap="gray", subtitles=['Magnitude', 'Phase'], toolbar_visible=False):
    ysize = array_m.shape[1]
    xsize = array_m.shape[2]
    figsize = 2 *(1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi
    def callback(z=None):
        fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        if z is None:
            ax[0].imshow(array_m,  interpolation=None, cmap=cmap)
            ax[1].imshow(array_p,  interpolation=None, cmap=cmap)
        else:
            ax[0].imshow(array_m[z, ...],  interpolation=None, cmap=cmap)
            ax[1].imshow(array_p[z, ...],  interpolation=None, cmap=cmap)
        ax[0].set_title(subtitles[0])
        ax[1].set_title(subtitles[1])
        fig.canvas.toolbar_visible = toolbar_visible
        if title:
            plt.suptitle(title)
            fig.canvas.manager.set_window_title(title)
        else:
            fig.canvas.header_visible = False
        if patch:
            for element in patch:
                new_patch_1 = copy(element)
                new_patch_2 = copy(element)
                ax[0].add_patch(new_patch_1)
                ax[1].add_patch(new_patch_2)
        plt.show()
    interact(callback, z=(0, array_m.shape[0] - 1))
# Show an image given its array (real number)
def show_array(array, title=None, patch=None, margin=0.1, dpi=80, cmap="gray", toolbar_visible=False):
    ysize = array.shape[1]
    xsize = array.shape[2]
    figsize = (1 + margin) * ysize / dpi, (1 + margin) * xsize / dpi
    def callback(z=None):
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_axes([margin, margin, 1 - 2 * margin, 1 - 2 * margin])
        if z is None:
            ax.imshow(array,  interpolation=None, cmap=cmap)
        else:
            ax.imshow(array[z, ...],  interpolation=None, cmap=cmap)
        fig.canvas.toolbar_visible = toolbar_visible
        if title:
            plt.suptitle(title)
            fig.canvas.manager.set_window_title(title)
        else:
            fig.canvas.header_visible = False
        if patch:
            for element in patch:
                new_patch = copy(element)
                ax.add_patch(new_patch)
        plt.show()
    interact(callback, z=(0, array.shape[0] - 1))
# Show an sitk image
def show_image(img, title=None, patch=None, margin=0.1, dpi=80, cmap="gray", toolbar_visible=False):
    array = sitk.GetArrayFromImage(img)
    show_array(array, title, patch, toolbar_visible=toolbar_visible)
# Show a pair of magnitude and phase sitk images
def show_image_pair(img_m, img_p, title=None, patch=None, margin=0.1, dpi=80, cmap="gray", subtitles=['Magnitude', 'Phase'], toolbar_visible=False):
    array_m = sitk.GetArrayFromImage(img_m)
    array_p = sitk.GetArrayFromImage(img_p)
    show_array_pair(array_m, array_p, title=title, patch=patch, subtitles=subtitles, toolbar_visible=toolbar_visible)
# Show a pair of magnitude and phase images from given complex array
def show_complex_array(array_comp, title=None, patch=None, margin=0.1, dpi=80, cmap="gray", subtitles=['Magnitude', 'Phase'], toolbar_visible=False):
    (array_m, array_p) = get_mag_phase_arrays(array_comp)
    show_array_pair(array_m, array_p, title=title, patch=patch, subtitles=subtitles, toolbar_visible=toolbar_visible)

##########################################################
## COMPLEX ARRAY MANIPULATION
# Return masked (non-complex) array
# output_pixel = value if mask_pixel == 0
# output_pixel = input_pixel if mask_pixel == 1
def array_mask(array, mask, value=0):
    foreground = np.multiply(array, mask)
    background = value*(np.ones_like(mask)-mask)
    masked_array = np.add(foreground, background)
    return masked_array
# Gets two pairs of complex arrays, and returns the pixel-wise ratio change,
# that is, divides the complex signals (deals with division by zero)
def complex_division(array1_c, array2_c, phase_shift, mask=None):
    # Phase shift compensation
    (array1_m, array1_p) = get_mag_phase_arrays(array1_c)
    (array2_m, array2_p) = get_mag_phase_arrays(array2_c)
    array2_p = array2_p + phase_shift        
    # Find indices where real parts of both arrays are zero
    zero_mag_indices = np.where((array1_m == 0) & (array2_m == 0))
    array1_m[zero_mag_indices] = 1.0
    array2_m[zero_mag_indices] = 1.0
    # Avoid division by zero incrementing array_2
    zero_mag_indices_array2 = np.where((array2_m == 0))
    array2_m[zero_mag_indices_array2] = 0.001 # Replace 0 by a small value
    # Calculate division result with conditions
    result_m = array1_m / array2_m
    result_p = array1_p - array2_p
    result = get_complex_array(result_m, result_p)
    # Apply mask if provided (mask with 1 in mag and 0 in phase)
    if mask is not None:
        result = array_mask(result, mask, 1)
    return result
# Get phase-shift between two complex array images. It is calculated as mean intensity change inside a mask
def get_phase_shift(array_1_c, array_2_c, mask):
    (array_1_m, array_1_p) = get_mag_phase_arrays(array_1_c)
    (array_2_m, array_2_p) = get_mag_phase_arrays(array_2_c)
    array_diff = array_1_p - array_2_p
    intensity_values = array_diff[mask==1]
    return np.mean(intensity_values) # Calculate the mean intensity values at the border mask
# From a complex array, get a pair of phase/mag arrays
def get_mag_phase_arrays(array_comp):
    array_m = np.absolute(array_comp)
    array_p = np.angle(array_comp)
    # Phase scaling
    p_max = np.max(array_p)
    p_min = np.min(array_p)
    if p_min < 0: # range [-pi, pi]
        array_p = array_p / p_max * np.pi
    else: # range [0, 2.*pi]
        array_p = array_p / p_max * 2.*np.pi
    return (array_m, array_p)
# From pair of mag/phase arrays, get complex array
def get_complex_array(array_m, array_p):
    # Scaling
    p_max = np.max(array_p)
    p_min = np.min(array_p)
    if p_min < 0: # range [-pi, pi]
        array_p = array_p / p_max * np.pi
    elif p_min >= 0: # range (0, 2.*pi]
        if p_max > 0:
            array_p = array_p / p_max * 2.*np.pi
    return (array_m * np.cos(array_p) + 1j * array_m * np.sin(array_p))
# Reconstruct a complex image array from an ITK image pair
# Return as a tuple of numpy array, origin, spacing and direction.
def image_to_complex_array(image_m, image_p):
    # Convert to numpy arrays
    array_m = sitk.GetArrayFromImage(image_m)
    array_p = sitk.GetArrayFromImage(image_p)
    array_comp = get_complex_array(array_m, array_p)
    # Note that the image loses the geometric information (i.e. origin, spacing, and direction),
    # once it is converted to a numpy array. The function returns them separately so that
    # a SimpleITK image can be reconstructed later. 
    origin = image_p.GetOrigin()
    spacing = image_p.GetSpacing()
    direction = image_p.GetDirection()
    return (array_comp, origin, spacing, direction)
def complex_array_to_image(array_comp, origin, spacing, direction, type=sitk.sitkFloat32):
    (array_m, array_p) = get_mag_phase_arrays(array_comp)
    image_m = sitk.GetImageFromArray(array_m, isVector=False)
    image_m.SetOrigin(origin)
    image_m.SetSpacing(spacing)
    image_m.SetDirection(direction)
    image_p = sitk.GetImageFromArray(array_p, isVector=False)
    image_p.SetOrigin(origin)
    image_p.SetSpacing(spacing)
    image_p.SetDirection(direction)
    return (image_m, image_p)
# Receives a pair of real/imag images
def mag_phase_images_to_real_imag(image_m, image_p):
    (array_comp, origin, spacing, direction) = image_to_complex_array(image_m, image_p)
    (array_r, array_i) = get_real_imag_arrays(array_comp)
    image_r = sitk.GetImageFromArray(array_r, isVector=False)
    image_r.SetOrigin(origin)
    image_r.SetSpacing(spacing)
    image_r.SetDirection(direction)
    image_i = sitk.GetImageFromArray(array_i, isVector=False)
    image_i.SetOrigin(origin)
    image_i.SetSpacing(spacing)
    image_i.SetDirection(direction)
    return(image_r, image_i)
# Receives a pair of real/imag arrays, and returns a pair of mag/phase arrays
def real_imag_to_mag_phase(array_r, array_i):
    array_comp = real_imag_to_complex_array(array_r, array_i)
    return get_mag_phase_arrays(array_comp)
# Receives a pair of real/imag arrays and returns a complex array
def real_imag_to_complex_array(array_r, array_i):
    return array_r + 1j * array_i
# Receives a pair of mag/phase array and returns real/imag arrays
def mag_phase_to_real_imag(array_m, array_p):
    array_comp = get_complex_array(array_m, array_p)
    return get_real_imag_arrays(array_comp)
# Receives a complex array and returns a pair of real/imag arrays
def get_real_imag_arrays(array_comp):
    array_r = array_comp.real
    array_i = array_comp.imag
    return(array_r, array_i)

##########################################################
## SITK IMAGE MANIPULATION
# Return string with the image direction canonical name
def getDirectionName(sitk_image):
    direction = sitk_image.GetDirection()
    if direction == AX_DIR:
        return 'AX'
    elif direction == SAG_DIR:
        return 'SAG'
    elif direction == COR_DIR:
        return 'COR'
    else:
        return 'Reformat'
# Scales the pixel intensity of image to values between a given interval (default = 0.0, 1.0)
def intensityScaleItk(sitk_input, min_output=0.0, max_output=1.0):
    # Use intensity rescaling filter
    intensityRescaleFilter = sitk.RescaleIntensityImageFilter()
    intensityRescaleFilter.SetOutputMaximum(max_output) #16 bit
    intensityRescaleFilter.SetOutputMinimum(min_output)
    return intensityRescaleFilter.Execute(sitk_input)
# Get mean intensity difference given two sitkImages
# Useful to calculate the phase shift between two phase images
def getMeanIntensityShift(sitkImage1, sitkImage2, mask):
    array =  sitk.GetArrayFromImage(sitkImage1 - sitkImage2)
    border_intensity_values = array[mask==1]
    return np.mean(border_intensity_values) # Calculate the mean intensity values at the border
# Return masked image
# output_pixel = 0 if mask_pixel == 0
# output_pixel = input_pixel if mask_pixel == 1
def maskImageItk(sitkImage, mask):
    sitkMasked = sitkImage * sitk.Cast(mask, sitkImage.GetPixelID())
    return sitkMasked
# Return sitk Image from numpy array
def numpyToItk(array, sitkReference, type=None):
    image = sitk.GetImageFromArray(array, isVector=False)
    if (type is None):
        image = sitk.Cast(image, sitkReference.GetPixelID())
    else:
        image = sitk.Cast(image, type)
    image.CopyInformation(sitkReference)
    return image
# Return blank itk Image with same information from reference volume
# Optionals: choose different pixelValue, type (pixel ID), choose different direction (and center image at the reference volume)
def createBlankItk(sitkReference, type=None, pixelValue=0, spacing=None, direction=None, center=True):
    image = sitk.Image(sitkReference.GetSize(), sitk.sitkUInt8)
    if (pixelValue != 0):
        image = pixelValue*sitk.Not(image)
    if (type is None):
        image = sitk.Cast(image, sitkReference.GetPixelID())
    else:
        image = sitk.Cast(image, type)  
    image.CopyInformation(sitkReference)
    if (direction is not None):
        image.SetDirection(direction)             # Set direction
        if center is True:
            moveToCenterItk(image, sitkReference) # Set origin
    if (spacing is not None):
        image.SetSpacing(spacing)                 # Set spacing
    return image
# Rotate an ITK image of angle (in degrees) along the slice axis and around image center with adjustable size so no data is lost
# By default the function uses a linear interpolator. For label images one should use the sitkNearestNeighbor interpolator so as not to introduce non-existant labels.
def resizableRotationItk(sitkImage, angle, defaultPixelValue=0.0, interpolator=sitk.sitkLinear):
    #Rotate around image center
    rotation_center = sitkImage.TransformContinuousIndexToPhysicalPoint([(sz-1)/2 for sz in sitkImage.GetSize()])
    direction = sitkImage.GetDirection()
    n = direction[6:9]
    cosTheta2 = math.cos(0.5*math.radians(angle))
    sinTheta2 = math.sin(0.5*math.radians(angle))
    rotation = sitk.VersorTransform([sinTheta2*n[0], sinTheta2*n[1], sinTheta2*n[2], cosTheta2], rotation_center)    
    #Compute the bounding box of the rotated volume
    start_index = [0,0,0]
    end_index = [sz-1 for sz in sitkImage.GetSize()]
    physical_corners = [sitkImage.TransformIndexToPhysicalPoint(corner) for corner in list(itertools.product(*zip(start_index, end_index)))]
    transformed_corners = [rotation.TransformPoint(corner) for corner in physical_corners] 
    transformed_corners_index = [sitkImage.TransformPhysicalPointToIndex(corner) for corner in transformed_corners] 
    min_index = np.min(transformed_corners_index,0).astype(int)
    max_index = np.max(transformed_corners_index,0).astype(int)
    # We resample onto an axis aligned grid, with origin defined by min_bounds, using
    # the original spacing.
    pixel_size = np.abs(np.array(max_index) - np.array(min_index)) + np.ones(3)
    new_size = [int(pixel_size[0]), int(pixel_size[1]), int(pixel_size[2])]
    origin = sitkImage.TransformIndexToPhysicalPoint([int(min_index[0]),int(min_index[1]),int(min_index[2])])
    return sitk.Resample(image1=sitkImage, size=new_size, transform=rotation, interpolator=interpolator,
                            outputOrigin=origin, outputSpacing=sitkImage.GetSpacing(), outputDirection=sitkImage.GetDirection(),
                            defaultPixelValue=defaultPixelValue, outputPixelType=sitkImage.GetPixelID())
# Resample image to a given spacing.
# If no spacing is provided, resample to isotropic pixels (using smallest spacing from original). 
# By default the function uses a linear interpolator. For label images one should use the sitkNearestNeighbor interpolator so as not to introduce non-existant labels.
def adjustSpacingItk(sitkImage, interpolator=sitk.sitkLinear, spacing=None):
    original_spacing = sitkImage.GetSpacing()
    original_size = sitkImage.GetSize()
    if (spacing is None):   # Make isotropic image
        if all(spc == original_spacing[0] for spc in original_spacing): # Image is already isotropic, just return a copy.
            return sitk.Image(sitkImage)        
        min_spacing = min(original_spacing)
        spacing = [min_spacing]*sitkImage.GetDimension()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, spacing)]
    return sitk.Resample(image1=sitkImage, size=new_size, transform=sitk.Transform(), interpolator=interpolator,
                         outputOrigin=sitkImage.GetOrigin(), outputSpacing=spacing, outputDirection=sitkImage.GetDirection(), 
                         defaultPixelValue=0.0, outputPixelType=sitkImage.GetPixelID())
# Change the origin of the sitk image so that its center matches with the center of a reference image
def setCenterItk(sitkImage, center, offset=[0,0,0]):
    image_center = getPhysicalCenterItk(sitkImage)
    total_offset = np.array(center) - np.array(image_center) - np.array(offset)
    offsetTransform = sitk.TranslationTransform(3, (total_offset[0], total_offset[1], total_offset[2]))
    image_origin = sitkImage.GetOrigin()
    new_origin = offsetTransform.TransformPoint(image_origin)
    sitkImage.SetOrigin(new_origin)
    return 
# Change the origin of the sitk image so that its center matches with the center of a reference image (DEPRECATED)
def moveToCenterItk(sitkImage, sitkReference, offset=[0,0,0]):
    reference_center = getPhysicalCenterItk(sitkReference) 
    return setCenterItk(sitkImage, reference_center, offset) 
# Get physical coordinates of the center of the volume
def getPhysicalCenterItk(sitkImage):
    center_index = [int((sz-1)/2) for sz in sitkImage.GetSize()]
    return sitkImage.TransformIndexToPhysicalPoint(center_index) 
# Return a copy of the provided sitk image with an added border with pixel value given by border_value
# Border dimensions are provided for each image dimension and should be provided in mm or px.
# The border_value can be a pixel value to fill the border, 
# or a tuple (min, max) to fill the border with random values in the given range (useful to fill image with noise)
def addBorderItk(sitkImage, border, border_value=0, dim_type='mm'):        
    # Border provided in px
    if dim_type=='px':
        border_px = border
    # Border provided in mm
    else:
        spacing = sitkImage.GetSpacing()
        border_px = [int(border[0]/spacing[0]), int(border[1]//spacing[1])]
    # Get copy of input image into a numpy array
    image_array = sitk.GetArrayFromImage(sitkImage)
    size = image_array.shape
    # Add border to numpy array
    if isinstance(border_value, int) or isinstance(border_value, float):
        # Border value is int or float
        image_array[:, :border_px[1], :] = border_value
        image_array[:, -border_px[1]:, :] = border_value
        image_array[:, :, :border_px[0]] = border_value
        image_array[:, :, -border_px[0]:] = border_value
    elif isinstance(border_value, tuple) and len(border_value)==2:
        # Border value is random value between two provided limits
        image_array[:, :border_px[1], :] = np.random.randint(border_value[0], border_value[1], size=(size[0],border_px[1],size[2]))
        image_array[:, -border_px[1]:, :] = np.random.randint(border_value[0], border_value[1], size=(size[0],border_px[1],size[2]))
        image_array[:, :, :border_px[0]] = np.random.randint(border_value[0], border_value[1], size=(size[0],size[1],border_px[0]))
        image_array[:, :, -border_px[0]:] = np.random.randint(border_value[0], border_value[1], size=(size[0],size[1],border_px[0]))
    # Get pixels from numpy array and return to sitk image
    sitkResult = sitk.Cast(sitk.GetImageFromArray(image_array), sitkImage.GetPixelID())
    sitkResult.SetSpacing(sitkImage.GetSpacing())
    sitkResult.SetDirection(sitkImage.GetDirection())
    sitkResult.SetOrigin(sitkImage.GetOrigin())
    return sitkResult

# Crop sitkImage in a give side (decreases the output image size by crop_size in px)
# while maintaining or specifying a new center.
# crop_size (int): The size of the crop in the specified direction (number of pixels)
# side (str): Direction to crop ('top', 'bottom', 'left', 'right').
def cropItk(sitkImage, crop_size, side='top'):
    # Get the image's attributes
    original_size = sitkImage.GetSize()
    # Calculate new size
    new_size = list(original_size)
    if side in ['top', 'bottom']:
        new_size = [original_size[0], original_size[1]-crop_size, original_size[2]]
    elif side in ['left', 'right']:
        new_size = [original_size[0]-crop_size, original_size[1], original_size[2]]
    else:
        print('Invalid side option')
        return None
    # Create a starting index for the crop
    start_index = [0,0,0]
    if side == 'top':
        start_index[1] = crop_size
    elif side == 'bottom':
        start_index[1] = original_size[1] - crop_size
    elif side == 'left':
        start_index[0] = crop_size
    elif side == 'right':
        start_index[0] = original_size[0] - crop_size
    else:
        print('Invalid side option')
        return None
    # Create a region of interest for cropping
    sitkNewImage = sitk.RegionOfInterest(sitkImage, size=new_size, index=start_index)
    return sitkNewImage

# Add padding to an sitkImage (increases the output image size by the padding value)
# The padding_value can be a pixel value to fill the padding, 
# or a tuple (min, max) to fill the padding with random values in the given range (useful to fill image with noise)
def addPaddingItk(sitkImage, padding, side='top', padding_value=0):
    # Get the original size and pixel type of the image
    original_size = sitkImage.GetSize()
    original_spacing = sitkImage.GetSpacing()
    original_direction = sitkImage.GetDirection()
    original_origin = sitkImage.GetOrigin()
    # Create a new empty image with the desired size and pixel type
    new_size = list(original_size)
    if side in ['top', 'bottom']:
        new_size[1] += padding
    elif side in ['left', 'right']:
        new_size[0] += padding
    else:
        print('Invalid side option')
        return None
    # Create new image with appropriate size and same spatial information
    sitkNewImage = sitk.Image(new_size, sitkImage.GetPixelID())
    # Convert the SimpleITK images to a numpy array
    image_array = sitk.GetArrayFromImage(sitkImage)
    new_image_array = sitk.GetArrayFromImage(sitkNewImage)
    # Copy the original pixels to the new padded image and complete the padding
    if isinstance(padding_value, int) or isinstance(padding_value, float):
        newPixel = padding_value
    elif isinstance(padding_value, tuple) and len(padding_value)==2:
        if (side == 'left') or (side == 'right'):
            newPixel = np.random.randint(padding_value[0], padding_value[1], size=(new_image_array.shape[0], new_image_array.shape[1], padding))
        else:
            newPixel = np.random.randint(padding_value[0], padding_value[1], size=(new_image_array.shape[0], padding, new_image_array.shape[2]))        
    else:
        print('Invalid padding_value')
        return None
    if side == 'top':
        new_image_array[:, padding:, :] = image_array
        new_image_array[:, :padding, :] = newPixel
    elif side == 'bottom':
        new_image_array[:, :original_size[1], :] = image_array
        new_image_array[:, original_size[1]:, :] = newPixel
    elif side == 'left':
        new_image_array[:, :, padding:] = image_array
        new_image_array[:, :, :padding] = newPixel
    elif side == 'right':
        new_image_array[:, :, :original_size[0]] = image_array
        new_image_array[:, :, original_size[0]:] = newPixel
    # Create final ITK padded image
    sitkNewImage = sitk.GetImageFromArray(new_image_array)
    sitk.Cast(sitkNewImage, sitkImage.GetPixelID())
    sitkNewImage.SetSpacing(original_spacing)
    sitkNewImage.SetDirection(original_direction)
    sitkNewImage.SetOrigin(original_origin)
    # Adjust origin for top and left paddings
    if side == 'top':
        b0 = np.array(sitkImage.TransformIndexToPhysicalPoint((0, original_size[1], 0)))
        b1 = np.array(sitkImage.TransformIndexToPhysicalPoint((0, new_size[1], 0)))
        new_origin = np.array(original_origin) - (b1-b0)
        sitkNewImage.SetOrigin(new_origin)
    elif side == 'left':
        b0 = np.array(sitkImage.TransformIndexToPhysicalPoint((original_size[0], 0, 0)))
        b1 = np.array(sitkImage.TransformIndexToPhysicalPoint((new_size[0], 0, 0)))
        new_origin = np.array(original_origin) - (b1-b0)
        sitkNewImage.SetOrigin(new_origin)          
    return sitkNewImage
# Return a flipped version of na sitk image, while keeping its spatial location
def flipImageItk(sitkImage, dir='vertical'):
    #Flip
    flipFilter = sitk.FlipImageFilter()
    if dir=='horizontal':
        flipFilter.SetFlipAxes([False,True,False])
    else:
        flipFilter.SetFlipAxes([True,False,False])
    sitkFlipped = flipFilter.Execute(sitkImage)
    # Adjust direction and origin to match original images
    sitkFlipped.SetDirection(sitkImage.GetDirection())
    sitkFlipped.SetOrigin(sitkImage.GetOrigin())
    return sitkFlipped
# Get the point physical coordinates of an image and flip the point vertically or horizontally
def getFlippedPoint(sitkReference, point_phys, dir='vertical'):
    # Get image dimensions
    imageSize = sitkReference.GetSize()
    # Convert the physical point to continuous index
    indexOrig = sitkReference.TransformPhysicalPointToIndex(point_phys)
    # Flip the index coordinates vertically
    if dir=='horizontal':
        indexFlip = [indexOrig[0], imageSize[1] - 1 - indexOrig[1], indexOrig[2]]
    else:
        indexFlip = [imageSize[0] - 1 - indexOrig[0], indexOrig[1], indexOrig[2]]
    # Convert the flipped index back to physical point
    flipped_point_phys = sitkReference.TransformIndexToPhysicalPoint(indexFlip)
    return flipped_point_phys

# Applies a Gaussian smoothing filter to an image, with conditional padding if necessary
# Inputs: 
#       sitkImage: The original image to which the Gaussian filter should be applied.
#       sigma: Float with standard deviation of the Gaussian smoothing filter.
# Output:
#       The smoothed image
def blurItk(sitkImage, sigma):
# Determine the minimum size required to avoid boundary issues.
    # This value will depend on the Gaussian sigma.
    min_size = max(4,int(2 * sigma + 1))
    # Calculate the padding needed for each dimension
    pad_lower = []
    pad_upper = []
    for dim in sitkImage.GetSize():
        if dim < min_size:
            padding_required = min_size - dim
            pad_lower.append(padding_required // 2)
            pad_upper.append(padding_required - (padding_required // 2))
        else:
            pad_lower.append(0)
            pad_upper.append(0)
    # Check if any padding is necessary
    needs_padding = any(pad > 0 for pad in pad_lower + pad_upper)
    if needs_padding:
        # Apply padding
        padded_img = sitk.ConstantPad(sitkImage, pad_lower, pad_upper, 0)
        # Apply Gaussian filter to padded image
        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetSigma(sigma)
        smoothed_padded_img = gaussian_filter.Execute(padded_img)
        # Crop back to original dimensions
        result_img = sitk.Crop(smoothed_padded_img, pad_lower, pad_upper)
    else:
        # Directly apply the Gaussian filter without padding
        gaussian_filter = sitk.SmoothingRecursiveGaussianImageFilter()
        gaussian_filter.SetSigma(sigma)
        result_img = gaussian_filter.Execute(sitkImage)
    return result_img