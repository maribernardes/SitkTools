# SitkUtils

**sitkMonaiIO:** implemented functions LoadSitkImage and PushSitkImage as MONAI transforms for loading and pushing sitk objects.
Based on the MONAI ITKReader and ITKWriter code.
- Tested with Coronal and Sagittal images for both single- and multi-slice volumes (TODO: check axial images)
- OBS: PushSitkImage still not tested without using EnsureChannelFirst and Orientation for adjusting tensor order to (C, D, H ,W) before. 
