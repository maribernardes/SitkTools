import random
import numpy as np
import math
import os
import json
from sitkUtils import *

##########################################################
## NEEDLE ARTIFACT UTILS
# Interpolate Artifacts: interpolate two needle artifacts at a given angle
# Takes the artifact's dataset path, a list of available artifact angles, and a desired angle (optional).
# If no angle is provided, a random angle is selected withing the available range
# Chooses the appropriate artifacts within the list and creates an interpolated artifact at the desired angle
def interpolateArtifacts(artifact_json_path, angle_list, target_angle=None):
    if not isinstance(angle_list[0], float):    # Make sure list contains floats
        angle_list = [float(item) for item in angle_list]
    angle_list = sorted(angle_list)             # Make sure list is ordered
    upper = max(angle_list)
    lower = min(angle_list)
    if target_angle is None:
        # Get random valid angle within available list
        target_angle = random.uniform(lower, upper)
    else:
         # Test is target angle is within available list
        if (target_angle>upper) or (target_angle<lower):
            raise ValueError('Target angle is not within the available range')
    # Select artifact templates           
    index = get_closest_index(target_angle, angle_list)
    if angle_list[index]>target_angle:
        index -= 1
    angle1 = angle_list[index]
    angle2 = angle_list[index+1]
    # Set weights for interpolation 
    delta1 = (target_angle-angle1)
    delta2 = (angle2-target_angle)
    delta = angle2-angle1
    w = [delta2/delta, delta1/delta]
    # Get needle artifact template        
    with open(artifact_json_path) as file_artifact:
        list_artifact = json.load(file_artifact)
    cols_artifact = list_artifact['columns']
    path = os.path.dirname(artifact_json_path)
    dir_artifact = os.path.join(path, list_artifact['directory'])
    artifact_image_list = np.array(list_artifact['images'], dtype=object)
    artifact_ang = artifact_image_list[:,cols_artifact.index('angle')] 
    artifact_type = artifact_image_list[:,cols_artifact.index('type')] 
    # Artifact 1
    angle = str(math.copysign(angle1, target_angle)) # Alows for negative 0.0 artifact
    iminfo_artifact1_m = artifact_image_list[(artifact_ang==str(angle)) & (artifact_type=='M'),:]
    iminfo_artifact1_p = artifact_image_list[(artifact_ang==str(angle)) & (artifact_type=='P'),:]
    artifact1_m = read_image(iminfo_artifact1_m, cols_artifact, dir_artifact)
    artifact1_p = read_image(iminfo_artifact1_p, cols_artifact, dir_artifact)
    center1 = np.array(iminfo_artifact1_m[0, cols_artifact.index('rot_center')])
    # Artifact 2
    angle = str(math.copysign(angle2, target_angle)) # Alows for negative 0.0 artifact
    iminfo_artifact2_m = artifact_image_list[(artifact_ang==str(angle)) & (artifact_type=='M'),:]
    iminfo_artifact2_p = artifact_image_list[(artifact_ang==str(angle)) & (artifact_type=='P'),:]
    artifact2_m = read_image(iminfo_artifact2_m, cols_artifact, dir_artifact)
    artifact2_p = read_image(iminfo_artifact2_p, cols_artifact, dir_artifact)
    center2 = np.array(iminfo_artifact2_m[0, cols_artifact.index('rot_center')])
    # Base and tip (artifact 1)
    artifactBase = np.array(iminfo_artifact1_m[0, cols_artifact.index('base')])
    artifactTip = np.array(iminfo_artifact1_m[0, cols_artifact.index('tip')])
    # Set rotation transform 1
    n = artifact1_m.GetDirection()[6:9]
    cosTheta = math.cos(0.5*math.radians(delta1))
    sinTheta = math.sin(0.5*math.radians(delta1))
    rotation = sitk.VersorTransform([sinTheta*n[0], sinTheta*n[1], sinTheta*n[2], cosTheta], center1)
    # Rotate artifact 1 and points
    rotated1_m = sitk.Resample(artifact1_m, rotation.GetInverse(), defaultPixelValue=1.0)
    rotated1_p = sitk.Resample(artifact1_p, rotation.GetInverse(), defaultPixelValue=0.0)
    rotatedBase = rotation.TransformPoint(artifactBase)
    rotatedTip = rotation.TransformPoint(artifactTip)  
    # Set rotation transform 2
    n = artifact2_m.GetDirection()[6:9]
    cosTheta = math.cos(0.5*math.radians(-delta2))
    sinTheta = math.sin(0.5*math.radians(-delta2))
    rotation = sitk.VersorTransform([sinTheta*n[0], sinTheta*n[1], sinTheta*n[2], cosTheta], center2)
    # Rotate artifact 2
    rotated2_m = sitk.Resample(artifact2_m, rotation.GetInverse(), defaultPixelValue=1.0)
    rotated2_p = sitk.Resample(artifact2_p, rotation.GetInverse(), defaultPixelValue=0.0)
    # Interpolate two templates           
    image1_m = rotated1_m * w[0]
    image1_p = rotated1_p * w[0]
    image2_m = rotated2_m * w[1]
    image2_p = rotated2_p * w[1]
    interpolated_m = image1_m + image2_m
    interpolated_p = image1_p + image2_p
    return (interpolated_m, interpolated_p, rotatedBase, rotatedTip, target_angle)
# Create Needle Labelmap: takes a given image and draws a white line at the provided base and tip physical positions
# If line width is omitted, use 1px width
# If drawOver is True, the cilinder is added on top of sitkReference, if False the cilinder is added to an empty image
def createLine(sitkReference, base, tip, radius_mm=1.0, drawOver=False, defaultPixelValue=255):
    # Get the image size and dimensions
    spacing = sitkReference.GetSpacing()
    size = sitkReference.GetSize()
    width, height, depth = size[0], size[1], size[2]
    # Create a numpy array from the image
    if drawOver is True:
        image_array = sitk.GetArrayFromImage(sitkReference)
    else:
        image_array = np.zeros((depth, height, width))
    # Convert start and end points to pixel coordinates
    baseIndex = sitkReference.TransformPhysicalPointToContinuousIndex(base)
    tipIndex = sitkReference.TransformPhysicalPointToContinuousIndex(tip)
    # Calculate the step sizes and number of steps
    dx = abs(tipIndex[0] - baseIndex[0])
    dy = abs(tipIndex[1] - baseIndex[1])
    dz = abs(tipIndex[2] - baseIndex[2])
    steps = max(dx, dy, dz)
    x_step = (tipIndex[0] - baseIndex[0]) / steps
    y_step = (tipIndex[1] - baseIndex[1]) / steps
    z_step = (tipIndex[2] - baseIndex[2]) / steps
    # Calculate the half line width
    radius_x = round(radius_mm / spacing[0])
    radius_y = round(radius_mm / spacing[1])
    radius_z = round(radius_mm / spacing[2])
    # Iterate over the line and set the intensity values to 255 (white)
    for i in range(round(steps+1)):
        x = round(baseIndex[0] + i * x_step)
        y = round(baseIndex[1] + i * y_step)
        z = round(baseIndex[2] + i * z_step)
        # Set the intensity values for pixels within the line width
        for w in range(-radius_x, radius_x + 1):
            for h in range(-radius_y, radius_y + 1):
                for d in range(-radius_z, radius_z + 1):
                    nx, ny, nz = x + w, y + h, z + d
                    if 0 <= nx < width and 0 <= ny < height and 0 <= nz < depth:
                        image_array[nz, ny, nx] = defaultPixelValue
    # Create a new SimpleITK image from the modified array
    sitkResult = sitk.Cast(sitk.GetImageFromArray(image_array), sitk.sitkUInt8)
    # Set the same origin, spacing, and direction as the input image
    sitkResult.SetOrigin(sitkReference.GetOrigin())
    sitkResult.SetSpacing(sitkReference.GetSpacing())
    sitkResult.SetDirection(sitkReference.GetDirection())
    return sitkResult

# Creates an image with same spatial information of given reference and draws a sphere at the provided center with given radius
# If drawOver is True, the sphere is added on top of sitkReference, if False the sphere is added to an empty image
def createSphere(sitkReference, center_phys, radius_mm, drawOver=False, defaultPixelValue=255):
    # Get spacial information from reference image
    spacing = sitkReference.GetSpacing()
    size = sitkReference.GetSize()
    width, height, depth = size[0], size[1], size[2]
    # Convert sphere center and radius to index space using the reference image
    center_index = sitkReference.TransformPhysicalPointToContinuousIndex(center_phys)
    radius = radius_mm / spacing[0]
    # Fill the pixels for the sphere
    if drawOver is True:
        array_sphere = sitk.GetArrayFromImage(sitkReference)
    else:
        array_sphere = np.zeros((depth, height, width))
    z, y, x = np.ogrid[:array_sphere.shape[0], :array_sphere.shape[1], :array_sphere.shape[2]]
    array_sphere[(x - center_index[0])**2 + (y - center_index[1])**2 + (z - center_index[2])**2 <= radius**2] = defaultPixelValue
    # Create a new SimpleITK image from the array
    sitkResult = sitk.Cast(sitk.GetImageFromArray(array_sphere), sitk.sitkUInt8)
    # Set the same origin, spacing, and direction as the input image
    sitkResult.SetOrigin(sitkReference.GetOrigin())
    sitkResult.SetSpacing(sitkReference.GetSpacing())
    sitkResult.SetDirection(sitkReference.GetDirection())
    return sitkResult

# Creates an image with same spatial information of given reference and draws a white cilinder at the provided center
# with given radius and height values
# If drawOver is True, the cilinder is added on top of sitkReference, if False the cilinder is added to an empty image
def createCilinder(sitkReference, center_phys, radius_mm, height_mm, drawOver=False, defaultPixelValue=255):
    # Get spacial information from reference image
    spacing = sitkReference.GetSpacing()
    size = sitkReference.GetSize()
    width, height, depth = size[0], size[1], size[2]
    # Create numpy array from the image
    if drawOver is True:
        arr = sitk.GetArrayFromImage(sitkReference)
    else:
        arr = np.zeros((depth, height, width))
    # Convert physical center to index
    center_index = sitkReference.TransformPhysicalPointToContinuousIndex(center_phys)
    # Convert radius and height from millimeters to pixels
    radius = radius_mm / spacing[0]
    height = height_mm / spacing[2]
    # Calculate slice indices that intersect with the cylinder's height
    z_start = max(round(center_index[2] - 0.5*height), 0)
    z_end = min(round(center_index[2] + 0.5*height), size[2] - 1)
    # Iterate through the slices that intersect with the cylinder's height
    for z in range(z_start, z_end+1):
        # Create a meshgrid for the slice
        xx, yy = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
        distance_to_center = np.sqrt((xx - center_index[0]) ** 2 + (yy - center_index[1]) ** 2)
        # Set pixels within the circle radius to defaultPixelValue
        arr[z, distance_to_center <= radius] = defaultPixelValue
    # Convert the modified numpy array back to an image
    sitkResult = sitk.Cast(sitk.GetImageFromArray(arr), sitk.sitkUInt8)
    sitkResult.SetOrigin(sitkReference.GetOrigin())
    sitkResult.SetSpacing(sitkReference.GetSpacing())
    sitkResult.SetDirection(sitkReference.GetDirection())
    return sitkResult

# Given a  a line start_point and end_point, find the line insersection point with the image boundary
def lineIntersectionWithBoundaries(sitkImage, start_point, end_point, num_points=100):
    # Create a set of equidistant points along the line segment
    t_values = np.linspace(0, 1, num_points)
    points = [tuple(start_point[i] + t * (end_point[i] - start_point[i]) for i in range(3)) for t in t_values]
    # Initialize 
    intersection_point = None
    for point in points:
        # Transform the point from spatial (physical) to index coordinates
        index = sitkImage.TransformPhysicalPointToIndex(point)
        # Check if the point is within the volume's bounds in index coordinates
        if all(0 <= index[i] < sitkImage.GetSize()[i] for i in range(3)):
            intersection_point = sitkImage.TransformIndexToPhysicalPoint(index)
        else:
            # This point is outside of the image volume
            break
    # Return the last point inside the image volume
    return intersection_point

# Given a line start_point and end_point, check if all points are inside a given mask volume
def isLineInsideMask(sitkMask, start_point, end_point, foreground_value=0, num_points=100):
    # Create a set of equidistant points along the line segment
    t_values = np.linspace(0, 1, num_points)
    points = [tuple(start_point[i] + t * (end_point[i] - start_point[i]) for i in range(3)) for t in t_values]
    # Loop all the points
    for point in points:
        # Transform the point from spatial (physical) to index coordinates
        index = sitkMask.TransformPhysicalPointToIndex(point)
        # Check if the point is within the volume's bounds in index coordinates
        if all(0 <= index[i] < sitkMask.GetSize()[i] for i in range(3)):
            # Check mask value
            if sitkMask[index]!=foreground_value:
                # This point is outside the mask
                return False
        else:
            # This point is outside of the image volume
            return False
    # All points inside the mask
    return True