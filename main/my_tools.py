#! /bin/python3

# Mauro Morais, 2021-12-22

# Imports
import os
import pickle
import json
import numpy as np
import pandas as pd
from functools import partial

# from scipy.fft import fft, ifft
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.color import rgb2lab
from skimage.exposure import rescale_intensity
from skimage.future import predict_segmenter
from skimage.feature import multiscale_basic_features
from skimage.segmentation import felzenszwalb, flood
from skimage.util import img_as_ubyte, img_as_uint
from sklearn.metrics import jaccard_score #, accuracy_score
from skimage.filters import threshold_local
from skimage.measure import regionprops_table, find_contours, shannon_entropy, regionprops
from skimage.filters import gaussian
from skimage.transform import rotate
from skimage.feature import graycoprops
from math import pi, degrees

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Image size adjustment function
def adjust_size(img, target_size=1024):
  
  # Adjust the size to target_size
  width = img.shape[1]
  height = img.shape[0]

  resize_factor = target_size / height
  
  new_height = round(height * resize_factor)
  new_width = round(width * resize_factor)
  new_shape = (new_height, new_width)

  new_img = resize(img, new_shape, anti_aliasing=True)

  return new_img 

# Image color adjustment
def adjust_color(img):
  '''
  Converts an RGB image to CIELab color space in 8-bit format. Using the 
  rgb2lab() function from skimage retunrs an 16-bit float image. This functions
  converts to CIELab and equalize the histogram.
  
  Parameters
  ----------
  img : (..., 3, ...) array like image
    The numpy image to convert to CIELab space.
    
  Returns
  -------
  out : numpy.ndarray
    The numpy image in Lab format (8-bit type).
  '''

  # Adjust the color image to CIELab space
  out = rgb2lab(img)

  # Return the lightness (L*) channel as float
  return out.astype('uint8') #[:, :, 0]

# Image pixel intensity adjustment
def adjust_intensity(img):

  # Get the 2 and 98 percentile values
  p2, p98 = np.percentile(img, (2, 98))
  img_eq = rescale_intensity(img, in_range=(p2, p98))
  # img_eq = equalize_hist(img)

  return img_eq

# Save the trained model
def save_model(model, path):
  fh = open(path, 'wb')
  pickle.dump(model, fh)
  fh.close()

  return 0

# Load a trained model
def load_model(path):
  fh = open(path, 'rb')
  model = pickle.load(fh)
  fh.close()

  return model

# Funtion to open JSON files
def read_json(filePath):
  fh = open(filePath, 'r', encoding="utf-8")
  ans = json.load(fh)
  return(ans)


# Function to save JSON files
def save_json(obj, filePath):
    with open(filePath, 'w+', encoding="utf-8") as fh:
      json.dump(obj, fh, indent=4, ensure_ascii=False)

# Filter the particles in the regionpros list by area size
# https://stackoverflow.com/questions/28586221/fast-filtering-of-a-list-by-attribute-of-each-element-of-type-scikit-image-regi
def filter_by_area(region_property, size=120):
    return(region_property.area > size)

def iou(y_true, y_pred):
    ji = round(jaccard_score(y_true.flatten(), y_pred.flatten(), pos_label=255), 4)
    return ji

# Function to extract the curvegram of a set of contourn points.
def curvegram(x, y):
  x1 = np.diff(x, append=x[0])
  x2 = np.diff(x1, append=x1[0])
  y1 = np.diff(y, append=y[0])
  y2 = np.diff(y1, append=y1[0])
  # dist = ((x1 ** 2) + (y1 ** 2)) ** (1/2)
  # perimeter = dist.sum()
  
  # j = np.complex(0,1)
  # f = 1/len(x)
  # X = fft(x)
  # x1 = ifft(j * 2 * pi * f * X).real
  # x2 = ifft( -((2 * pi * f) ** 2) * X ).real
  
  # Y = fft(y)
  # y1 = ifft(j * 2 * pi * f * Y).real
  # y2 = ifft( -((2 * pi * f) ** 2 * Y) ).real
  
  num = (x1 * y2) - (x2 * y1)
  den = ( (x1 ** 2) + (y1 ** 2) ) ** (3/2)
  den[-1] = 1 # to avoid division by zero
  curve = num / den 
   
  return np.abs(curve)#, perimeter

# Function to calculate the bending energy of the curvegram
def bending_energy(curve, perimeter):
  num = (curve ** 2).sum()
  ans = num / perimeter
  return ans

# Function to flip an 2D image vertically with respect to line_index
# line_index is the horizontal line of the flip
def flip_image(img, line_index):
  
  new_img = np.zeros(img.shape)

  index = 0

  while(line_index - index >= 0 and line_index + index < img.shape[0]):
    index_above = line_index - index
    index_below = line_index + index

    for x in range(img.shape[1]):
      new_img[index_above, x] = img[index_below, x]
      new_img[index_below, x] = img[index_above, x]

    index += 1

  return new_img

# Function to calculate the color level co-occurence matrix
# between two channels. It assumes the color componentes similarities
# are in the same position of the image between different color channels
# W. Gui et al., Minerals Engineering, 46-47 (2013) 60-67.
def color_comatrix(channelA, channelB, levels=256):
  height, width = channelA.shape
  output = np.zeros((levels,levels, 1, 1), np.uint32)

  for x in range(width):
    for y in range(height):
      valA = channelA[y, x]
      valB = channelB[y, x]
      output[valA, valB, 0, 0] += 1
  
  return output

## Options for segmentation models model
### Parameter for local threhsolding
ws = 21

### sk-learn RF model
nCPU = os.cpu_count() - 1
sigma_min = 0.5
sigma_max = 10
num_sigma = 7
model_path = '../data/fields/models/field0008_model.pkl'

### Felzenszwalb parameters
sigma_fz = 0.5
min_size = 100
scale_fz = 250

### Other options
channel_axis = None

# Feature extraction function
features_func = partial(multiscale_basic_features,
                        intensity=True, edges=True, texture=True,
                        sigma_min=sigma_min, sigma_max=sigma_max,
                        num_sigma=num_sigma,
                        num_workers=nCPU, channel_axis=channel_axis)

# Function to segment the image
def segment_image(image, method='local_mean', window_size=ws, model_path=None, scale=scale_fz):
  # Local mean thresholding
  if method == 'local_mean':
    thresh = threshold_local(image[..., 0], block_size=window_size, method='mean')
    image_segmented = image > thresh
  
  # scikit-learn model
  elif method == 'sklean_rf':
    if model_path == None:
      return "sklearn_rf method requires a pretrained model."
    else: seg_model = load_model(model_path)

    features = features_func(image[..., 0])
    image_segmented = predict_segmenter(features, seg_model)
  
  # Felzenszwalb method
  elif method == 'felzen':
    image_segmented = felzenszwalb(image, scale=scale_fz, sigma=sigma_fz, min_size=min_size)
  
  return image_segmented

## Binarization parameters
# images_info_path = "../results/images_info.csv"
# images_info = pd.read_csv(images_info_path)
# 
# objects = ["NUCLEUS", "NUCLEUS_CELL_BODY", "T_CRUZI"]
# position_data_path = '../results/position_data.csv'
# position_data = pd.read_csv(position_data_path)
# position_data = position_data[position_data.structure.isin(objects)]

# Function to binarize the image
# def binarize_image(image, images_info=images_info, file_name='field0001.jpg', position_data=position_data):#, filter_objects=objects):
#   # Get the resize factor of the given image
#   resize_factor = images_info[images_info.filename == file_name].resize_factor[0]
# 
#   # Get the postion data
#   positions = position_data[position_data.filename == file_name]
#   # positions = positions[positions.structure.isin(filter_objects)]
# 
#   x = np.array(positions.x) * resize_factor
#   y = np.array(positions.y) * resize_factor
# 
#   segmentation = np.zeros_like(image)
# 
#   for i in range(len(x)):
#     roi = flood(image, (int(x[i]), int(y[i])))
#     segmentation += roi
#     image_binarized = segmentation >= 1
# 
#   return img_as_ubyte(image_binarized)

# Function to extract features
def feature_extractor(img, img_label):
  """Calculate object's properties and return them as a pandas table.
  TODO: We need to refactor this function!

  Parameters
  ----------
  img : (N, M, C) ndarray
    Intensity input image.
  img_label : (N, M) ndarray
    Labeled input image with the same size as img. Labels with value 0 are ignored.

  Returns
  -------
  out_data : pandas.DataFrame
    The output DataFrame with features of the objects.
  """
  ## Get the label of each region (object)
  # img_label = label(img_bin)

  ## Define the properties to extract
  props = [
    "label",
    "area",
    "perimeter",
    "axis_major_length",
    "axis_minor_length",
    "orientation",
    "centroid",
    # "centroid_local",
    "centroid_weighted_local", # centroid_weighted_local-[y,x]-[r,g,b]
    "intensity_max",
    "intensity_min",
    "intensity_mean",
    "moments_hu",
    "image_filled",
    "image_intensity"]

  ## Extract the properties (features) of each region
  out_dict = regionprops_table(
    img_label,
    img, 
    properties=props)

  #### Calculate extra properties ####
  #### Find region contours and curvegrams ####
  region_contours = []
  curves = []
  for region in out_dict['image_filled']:
    region_ = np.pad(region, 1)
    contour = find_contours(region_)
    x = contour[0][:, 1]
    y = contour[0][:, 0]
    curve = curvegram(x, y)
    curve_smooth = gaussian(curve, sigma=1)

    region_contours.append(contour)
    curves.append(curve_smooth)

  # Add region contour and curvegrams to out_dict
  out_dict['contours'] = region_contours
  out_dict['curvegrams'] = curves

  #### Calculate the curvegram's metrics ####
  curve_entropy = []
  curve_variance = []
  curve_bending_energy = []
  for i, curve in enumerate(out_dict['curvegrams']):
    var = np.var(curve)
    ent = shannon_entropy(curve)
    ben = bending_energy(curve, out_dict['perimeter'][i])

    curve_variance.append(var)
    curve_entropy.append(ent)
    curve_bending_energy.append(ben)

  # Add the curvegram metrics to out_dict
  out_dict['curve_variance'] = curve_variance
  out_dict['curve_entropy'] = curve_entropy
  out_dict['curve_bending_energy'] = curve_bending_energy

  #### Calculate geometric descriptors ####
  max_dist_boundary_centroid = []
  min_dist_boundary_centroid = []
  mean_dist_boundary_centroid = []

  aspect_ratio = []
  circularity = []
  thickness_ratio = []
  major_perimeter_ratio = []

  for i, contour in enumerate(out_dict['contours']):
    # Calculate the distances from centroid to contour
    center_x = out_dict['centroid_weighted_local-1-0'][i]
    center_y = out_dict['centroid_weighted_local-0-0'][i]
    dx = center_x - contour[0][:,1]
    dy = center_y - contour[0][:,0]
    centroid_contour_distances = ((dx ** 2) + (dy ** 2)) ** 0.5

    major = out_dict['axis_major_length'][i]
    minor = out_dict['axis_minor_length'][i]
    area = out_dict['area'][i]
    perimeter = out_dict['perimeter'][i]

    # calculate geometric descriptors
    ar = major / minor
    circ = 4 * pi * (area / (perimeter ** 2))
    tr = (perimeter ** 2) / area
    mpr = major / perimeter

    max_dist = centroid_contour_distances.max()
    min_dist = centroid_contour_distances.min()
    mean_dist = centroid_contour_distances.mean()

    max_dist_boundary_centroid.append(max_dist)
    min_dist_boundary_centroid.append(min_dist)
    mean_dist_boundary_centroid.append(mean_dist)

    aspect_ratio.append(ar)
    thickness_ratio.append(tr)
    major_perimeter_ratio.append(mpr)
    circularity.append(circ)

  # Add the centroid distances metrics to out_dict
  out_dict['max_dist_boundary_centroid'] = max_dist_boundary_centroid
  out_dict['min_dist_boundary_centroid'] = min_dist_boundary_centroid
  out_dict['mean_dist_boundary_centroid'] = mean_dist_boundary_centroid

  # Add the geometric metrics to out_dict
  out_dict['aspect_ratio'] = aspect_ratio
  out_dict['thickness_ratio'] = thickness_ratio
  out_dict['major_perimeter_ratio'] = major_perimeter_ratio
  out_dict['circularity'] = circularity

  ### Calculate bilateral symmetry ###
  symmetry = []

  for i, region in enumerate(out_dict['image_filled']):
    # Get the center of mass and orientation angle (in radians)
    angle = out_dict['orientation'][i]
    center_x = out_dict['centroid_weighted_local-1-0'][i]
    center_y = out_dict['centroid_weighted_local-0-0'][i]

    # Convert radians to degrees
    angle_deg = degrees(angle)

    # Rotate the region
    region_rot = rotate(region, -angle_deg, resize=True, 
      center=(center_x, center_y))
    region_rot = region_rot.astype('uint8')

    # Flip the region
    region_rot_flip = flip_image(region_rot, int(center_y))

    # Calculate symmetry
    n2 = np.logical_and(region_rot, region_rot_flip)
    n = np.logical_or(region_rot, region_rot_flip)
    sym = round(n2.sum() / n.sum(), 4)
    symmetry.append(sym)

  out_dict['bilateral_symmetry'] = symmetry

  ### Calculate color descriptors ###
  colour_mean = []
  colour_median = []
  colour_mode = []
  colour_range = []
  colour_variance = []

  for i, region in enumerate(out_dict['image_intensity']):
    mask = out_dict['image_filled'][i]

    mean = np.mean(region[mask])
    median = np.median(region[mask])

    vals, counts = np.unique(region[mask], return_counts=True)
    mode_idx = np.argwhere(counts == np.max(counts))
    mode = vals[mode_idx].flatten()[0]

    range_vals = np.max(region[mask]) - np.min(region[mask])
    variance = np.var(region[mask])

    colour_mean.append(mean)
    colour_median.append(median)
    colour_mode.append(mode)
    colour_range.append(range_vals)
    colour_variance.append(variance)

  out_dict['colour_mean'] = colour_mean
  out_dict['colour_median'] = colour_median
  out_dict['colour_mode'] = colour_mode
  out_dict['colour_range'] = colour_range
  out_dict['colour_variance'] = colour_variance

  ### Calculate the color co-occurence matrix metrics ###
  ccm_ll_entropy = []
  ccm_la_entropy = []
  ccm_lb_entropy = []
  ccm_aa_entropy = []
  ccm_ab_entropy = []
  ccm_bb_entropy = []

  ccm_ll_asm = []
  ccm_la_asm = []
  ccm_lb_asm = []
  ccm_aa_asm = []
  ccm_ab_asm = []
  ccm_bb_asm = []

  ccm_la_contrast = []
  ccm_lb_contrast = []
  ccm_ab_contrast = []

  ccm_la_idm = []
  ccm_lb_idm = []
  ccm_ab_idm = []

  ccm_la_corr = []
  ccm_lb_corr = []
  ccm_ab_corr = []

  for region in out_dict['image_intensity']:

    # Get the channel array
    ch1 = region[:,:,0]
    ch2 = region[:,:,1]
    ch3 = region[:,:,2]

    # Calculate the colour co-occurence matrices
    ccm_ll = color_comatrix(ch1, ch1)
    ccm_la = color_comatrix(ch1, ch2)
    ccm_lb = color_comatrix(ch1, ch3)
    ccm_aa = color_comatrix(ch2, ch2)
    ccm_ab = color_comatrix(ch2, ch3)
    ccm_bb = color_comatrix(ch3, ch3)

    # Calculate entropy
    ccm_ll_entropy.append(shannon_entropy(ccm_ll))
    ccm_la_entropy.append(shannon_entropy(ccm_la))
    ccm_lb_entropy.append(shannon_entropy(ccm_lb))
    ccm_aa_entropy.append(shannon_entropy(ccm_aa))
    ccm_ab_entropy.append(shannon_entropy(ccm_ab))
    ccm_bb_entropy.append(shannon_entropy(ccm_bb))

    # Calculate angular second moment (ASM)
    ccm_ll_asm.append(graycoprops(ccm_ll, prop='ASM')[0][0])
    ccm_la_asm.append(graycoprops(ccm_la, prop='ASM')[0][0])
    ccm_lb_asm.append(graycoprops(ccm_lb, prop='ASM')[0][0])
    ccm_aa_asm.append(graycoprops(ccm_aa, prop='ASM')[0][0])
    ccm_ab_asm.append(graycoprops(ccm_ab, prop='ASM')[0][0])
    ccm_bb_asm.append(graycoprops(ccm_bb, prop='ASM')[0][0])

    # Calculate the contrast
    ccm_la_contrast.append(graycoprops(ccm_la, prop='contrast')[0][0])
    ccm_lb_contrast.append(graycoprops(ccm_lb, prop='contrast')[0][0])
    ccm_ab_contrast.append(graycoprops(ccm_ab, prop='contrast')[0][0])

    # Calculate the inverse difference moment (IDM, homogeneity)
    ccm_la_idm.append(graycoprops(ccm_la, prop='homogeneity')[0][0])
    ccm_lb_idm.append(graycoprops(ccm_lb, prop='homogeneity')[0][0])
    ccm_ab_idm.append(graycoprops(ccm_ab, prop='homogeneity')[0][0])

    # Calculate the correlation
    ccm_la_corr.append(graycoprops(ccm_la, prop='correlation')[0][0])
    ccm_lb_corr.append(graycoprops(ccm_lb, prop='correlation')[0][0])
    ccm_ab_corr.append(graycoprops(ccm_ab, prop='correlation')[0][0])

  # Add ccm metrics to out_dict
  out_dict['ccm_ll_entropy'] = ccm_ll_entropy
  out_dict['ccm_la_entropy'] = ccm_la_entropy
  out_dict['ccm_lb_entropy'] = ccm_lb_entropy
  out_dict['ccm_aa_entropy'] = ccm_aa_entropy
  out_dict['ccm_ab_entropy'] = ccm_ab_entropy
  out_dict['ccm_bb_entropy'] = ccm_bb_entropy

  out_dict['ccm_ll_asm'] = ccm_ll_asm
  out_dict['ccm_la_asm'] = ccm_la_asm
  out_dict['ccm_lb_asm'] = ccm_lb_asm
  out_dict['ccm_aa_asm'] = ccm_aa_asm
  out_dict['ccm_ab_asm'] = ccm_ab_asm
  out_dict['ccm_bb_asm'] = ccm_bb_asm

  out_dict['ccm_la_contrast'] = ccm_la_contrast
  out_dict['ccm_lb_contrast'] = ccm_lb_contrast
  out_dict['ccm_ab_contrast'] = ccm_ab_contrast

  out_dict['ccm_la_idm'] = ccm_la_idm
  out_dict['ccm_lb_idm'] = ccm_lb_idm
  out_dict['ccm_ab_idm'] = ccm_ab_idm

  out_dict['ccm_la_corr'] = ccm_la_corr
  out_dict['ccm_lb_corr'] = ccm_lb_corr
  out_dict['ccm_ab_corr'] = ccm_ab_corr

  ##### Build the output DataFrame #####
  out_data = pd.DataFrame({
    # 'file_name': [file for x in out_dict['label']],
    'label': out_dict['label'],
    'centroid_x': out_dict['centroid-1'],
    'centroid_y': out_dict['centroid-0'],
    'area': out_dict['area'],
    'perimeter': out_dict['perimeter'],
    'ccm_ll_entropy': out_dict['ccm_ll_entropy'],
    'ccm_la_entropy': out_dict['ccm_la_entropy'],
    'ccm_lb_entropy': out_dict['ccm_lb_entropy'],
    'ccm_aa_entropy': out_dict['ccm_aa_entropy'],
    'ccm_ab_entropy': out_dict['ccm_ab_entropy'],
    'ccm_bb_entropy': out_dict['ccm_bb_entropy'],
    'ccm_ll_asm': out_dict['ccm_ll_asm'],
    'ccm_la_asm': out_dict['ccm_la_asm'],
    'ccm_lb_asm': out_dict['ccm_lb_asm'],
    'ccm_aa_asm': out_dict['ccm_aa_asm'],
    'ccm_ab_asm': out_dict['ccm_ab_asm'],
    'ccm_bb_asm': out_dict['ccm_bb_asm'],
    'ccm_la_contrast': out_dict['ccm_la_contrast'],
    'ccm_lb_contrast': out_dict['ccm_lb_contrast'],
    'ccm_ab_contrast': out_dict['ccm_ab_contrast'],
    'ccm_la_idm': out_dict['ccm_la_idm'],
    'ccm_lb_idm': out_dict['ccm_lb_idm'],
    'ccm_ab_idm': out_dict['ccm_ab_idm'],
    'ccm_la_corr': out_dict['ccm_la_corr'],
    'ccm_lb_corr': out_dict['ccm_lb_corr'],
    'ccm_ab_corr': out_dict['ccm_ab_corr'],
    'minor_axis': out_dict['axis_minor_length'],
    'major_axis': out_dict['axis_major_length'],
    'circularity': out_dict['circularity'],
    'thickness_ratio': out_dict['thickness_ratio'],
    'aspect_ratio': out_dict['aspect_ratio'],
    'major_axis_perimeter_ratio': out_dict['major_perimeter_ratio'],
    'max_dist_boundary_centroid': out_dict['max_dist_boundary_centroid'],
    'min_dist_boundary_centroid': out_dict['min_dist_boundary_centroid'],
    'mean_dist_boundary_centroid': out_dict['mean_dist_boundary_centroid'],
    'bilateral_symmetry': out_dict['bilateral_symmetry'],
    'colour_mean': out_dict['colour_mean'],
    'colour_median': out_dict['colour_median'],
    'colour_mode': out_dict['colour_mode'],
    'colour_range': out_dict['colour_range'],
    'colour_variance': out_dict['colour_variance'],
    'moments_hu-0': out_dict['moments_hu-0'],
    'moments_hu-1': out_dict['moments_hu-1'],
    'moments_hu-2': out_dict['moments_hu-2'],
    'moments_hu-3': out_dict['moments_hu-3'],
    'moments_hu-4': out_dict['moments_hu-4'],
    'moments_hu-5': out_dict['moments_hu-5'],
    'moments_hu-6': out_dict['moments_hu-6'],
    'curvature_variance': out_dict['curve_variance'],
    'curvature_entropy': out_dict['curve_entropy'],
    'curvature_energy': out_dict['curve_bending_energy'],
  })

  return out_data

# Felzenszwalb's segmentation function
def felzen_segmentation(img, scale=250, sigma=0.5, min_size=100):
    '''Function to segment RGB image using the Felzenszwalb algorithm from 
    skimage implementation.

    Parameters
    ----------
    img : (width, height, 3) ndarray
        The input image for segmentation.
    scale : float
        The Felzenszwalb's scale parameter. Defines the size of the segments.
        Higher means larger clusters.
    sigma : float
        Preprocessing Gaussian smoothing filter size.
    min_size : int
        Minimum component size. Enforced using post processing.
    
    Returns
    -------
    segments_fz : (width, height) ndarray
        The output uint 16-bit image with segments label.
    '''
    segments_fz = felzenszwalb(img, scale=scale, sigma=sigma, min_size=min_size)
    return img_as_uint(segments_fz, force_copy=True)

#TODO: Add documentation for functions

# Plot function
def plot_pair(imgA, imgB):
    '''Function to plot two images for comparisson.
    
    '''
    if imgA.shape != 3 : colsA = 'gray'
    if imgB.shape != 3 : colsB = 'gray'

    fig, ax = plt.subplots(1, 2, figsize=(10, 10), sharex=True, sharey=True)
    ax[0].imshow(imgA, cmap=colsA)
    ax[0].set_title(f"imgA")

    ax[1].imshow(imgB, cmap=colsB)
    ax[1].set_title(f"imgB")
    # ax[1].scatter(x,y, color='red')

    for a in ax: a.set_axis_off()

    plt.tight_layout()
    plt.show()



# Get the properties and remove big areas
def remove_big_areas(segments, max_area = 3000):
  '''
  Function to remove big segments in Felzenswalbs segmented image.
  It takes segments with more than max_area and convert its pixels values to
  zero.

  Parameters
  ----------
  segments : numpy.ndarray
    A numpy array of segmented image. Each region correspond to the 
    Felzenswalbs segment.
  max_area : int
    An integer representing the size of the areas that will be removed. Segments 
    bigger than this value will be converted to zero.
  
  Returns
  -------
  segments : numpy.ndarray
    The segmented image, but areas bigger than max_areas pixels contains zero value.
  '''
  ## Get the region properties
  segments_props = regionprops(segments)
  ## Loop each region
  for props in segments_props:
      ### Remove segments with area bigger than max_area
      if props.area >= max_area: 
          segment_label = props.label
          segments[segments == segment_label] = 0

  return segments

# Get the positions of the parasites in the image
def get_positions(positions):
  '''
  Function to get the position of each parasite in an image.
  
  Parameters
  ----------
  position : pandas.DataFrame
    A DataFrame with the position of each parasite for the current image in
    new_x and new_y series.
    
  Returns
  -------
  x : numpy.array
    An array with the X's coordinates of the parasites.
  y : numpy.array
    An array with the Y's coordinates of the parasites.
  '''

  x = np.array(positions.new_x) 
  y = np.array(positions.new_y)
  return x, y

# Crop the parasite ROI function
def crop_roi(img, x, y, size = 50):
  '''
  Function to crop a single parasite from image located at (x,y).
  
  Parameters
  ----------
  img : numpy.ndarray
    A numpy array of the image.
  x : int
    The X position of the parasite.
  y : int
    The Y position of the parasite.
  size : int
    The lateral size of the crop.
  
  Returns
  -------
  crop : np.ndarray
    A numpy array of the cropped region with shape (size*2, size*2)
  
  NOTE: If the parasite is at the border of the image, the crop size will be
  smaller.

  '''
  # Check if parasite is at border
  if x-size <= 0:
    crop = img[y-size:y+size, 0:x+size]
  elif x+size >= img.shape[1]:
    crop = img[y-size:y+size, x-size:img.shape[1]]
  else:
    crop = img[y-size:y+size, x-size:x+size]
    
  return crop

# Crop and extract features of all parasites in an image
def extract_features(img, segments, x, y):
  '''
  Extract the features from all segments in a 100 x 100 pixel^2 area, 
  cetered at (x,y). It uses the manualy annotated positions of the parasites.

  Returns
  -------
  features_data : pandas.DataFrame
    The DataFrame with the features of the segments inside of the cropped
    location.
  '''
  features = []
  i=0

  ## Loop each parasite
  for cc, rr in zip(x, y):
      ### Crop the images
      img_crop = crop_roi(img, cc, rr)
      seg_crop = crop_roi(segments, cc, rr)

      ### Extract the features
      df = feature_extractor(img_crop, seg_crop)

      ### Get the parasite label
      parasite_label = seg_crop[50,50]
      parasite_ID = "parasite_" + f'{i:03}'

      ### Check wether it is there or not
      if parasite_label != 0: df['structure'] = df.label == parasite_label
      else: df['structure'] = False
      df.insert(0, "parasite_ID", parasite_ID)

      ### Add the DataFrame
      features.append(df)

      ### Count the number of parasites for ID
      i += 1
    
    ## Combine the dataframes
  features_data = pd.concat(features)

  return features_data

# Save the features dataframe
def save_features_data(file_path, features_data):
  '''
  Save the features extracted for the current image.

  Parameters
  ----------
  file_path : str
    The path to the file where data will be saved.
  features_data : pandas DataFrame
    The data frame with features of the objects of the current image.
  '''

  features_data.to_csv(file_path, index=False)

# Get the position of all parasites in all images
def get_parasite_positions_data(positions_file_path):
  '''
  Remove unwanted objects from the annotation dataset.

  Parameters
  ----------
  positions_file_path : str
    A string to the CSV file containing the postion of each parasite.
  
  Return
  ------
  position_data : pandas.DataFrame
    The pandas DataFrame objects containing only the position of the parasites. 
    KINETOPLAST and KINETOPLAST_CELL_BODY positions are removed.
  '''
  # Objects to filter from original data
  filter_objects = ["NUCLEUS", "NUCLEUS_CELL_BODY", "T_CRUZI"]

  # Get the annotated parasites
  position_data = pd.read_csv(positions_file_path)
  position_data = position_data[position_data.structure.isin(filter_objects)]

  return position_data

# To convert positions into yolo coords.
def convert_to_yolo_coords(positions,
    img_width = 768, img_height = 1024, 
    box_width = 50, box_height= 50):
    '''
    Functions to convert the annotated coordinates in he images to YOLO format.
    '''
    # Set the positions for YOLO
    positions['yolo_x'] = positions['new_x'] / img_width
    positions['yolo_y'] = positions['new_y'] / img_height
    positions['yolo_width'] = box_width / img_width
    positions['yolo_height'] = box_height / img_height

    return positions

# Function to process all the images
def process_all(img_path, img_files, positions_file):
  '''
  Function to process all the images and return the list of extracted 
  features.

  Parameters
  ----------
  img_path : str
    A string to the folder with the images to be processed.
  img_files : list of str
    A list of strings with the images file names.
  position_file : str
    A string path to the ground truth positions of the parasites.

  Returns
  -------
  features_list : list of pandas.DataFrame
    A list of pandas.DataFrames containg the features of segmented 
    parasite.
  unknown_list : list of pandas.DataFrame
    A list of pandas.DataFrame containg the features of unknown
    objects.
    
  '''
  # A list of pandas DataFrames
  features_list = []
  unknown_list = []

  for file in img_files:
    ## Read the image
    img = imread(os.path.join(img_path, file))
    # img = adjust_size(img)

    ## Convert to CIELab space
    img_lab = adjust_color(img)

    ## Get the position annotation for the current file
    positions_data = get_parasite_positions_data(positions_file)
    positions = positions_data[positions_data.filename == file]

    ## Segment the image using Felzenszwalb algorithm
    segments_fz = felzen_segmentation(img)

    ## Remove segments with area >= max_area pixels
    segments_fz = remove_big_areas(segments_fz)

    ## Get x,y positions of the parasites in the current image
    x, y = get_positions(positions)

    ## Get the features of each parasite in an image
    features_data = extract_features(img_lab, segments_fz, x, y)
    features_data.insert(0, "filename", file)

    ## Remove parasites that were not segmented 
    ## and select features of unknown objetcs
    features_parasite = features_data[features_data['structure'] == True]
    features_unknown = features_data[features_data['structure'] == False]


    ## Add data frame to features list
    features_list.append(features_parasite)
    unknown_list.append(features_unknown)

    ## Retunr the list of features
  return features_list, unknown_list

# Function to fit a set of models
def fit_models(x, y, models):
  '''
  This functions uses the sklearn model fitting method to fit a list of predictors to data points.

  Parameters
  ----------
  x : pandas.DataFrame or numpy.ndarray
    The input feature space training data.
  y : pandas.series or numpy.ndarray
    The input target with class labels.
  models : list
    A list of instantiated models
  
  Returns
  -------
  models
    Fitted models with training data
  '''
  for clf in models:
    clf.fit(x, y)

# Get the predictions of a fitted model
def get_predictions(x, clf, probability=True):
  predictions = clf.predict(x)

  if probability: probabilities = clf.predict_proba(x)[:,1]
  else: probabilities = None

  return predictions, probabilities

# Get the prediction metrics
def get_metrics(y, predictions, probabilities):
    
  fpr, tpr, _ = roc_curve(y, probabilities)
  
  conf_matrix = confusion_matrix(y, predictions)
  report = classification_report(y, predictions, digits=4)

  return (fpr, tpr, conf_matrix, report)

# Plot ROC curve
def plot_roc_curve(fpr, tpr, model):

  auc_score = auc(fpr, tpr)

  plt.figure(figsize=(8,6))
  plt.grid(color = 'lightgray', linestyle = '--', linewidth = 1)
  plt.plot([0,1], [0,1], color='gray', lw=1, linestyle='--')
  plt.plot(fpr, tpr, color='red', lw=2, label="ROC curve AUC: %0.2f" % auc_score)
  
  plt.title("Receiver operating characteristic %s" % model.__class__.__name__)
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.legend(loc="lower right")
  plt.show()

# Calculate the model performance metrics based on confusion matrix.
def calculate_performance(clf, conf_matrix, fpr, tpr):
    model_name = clf.__class__.__name__
    tn, fp, fn, tp = conf_matrix.ravel()
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    pre = tp / (tp + fp)
    f1_score = 2 * (sen * pre) / (sen + pre)
    acc_score = (tp + tn) / (tn + tp + fp + fn)
    auc_score = auc(fpr, tpr)

    results = {
      "Model": model_name,
      "Sensitivity": round(sen, 4),
      "Specificity": round(spe, 4),
      "Precision": round(pre, 4),
      "Accuracy": round(acc_score, 4),
      "F1-score": round(f1_score, 4),
      "AUC": round(auc_score, 4),
      "FPR": fpr.tolist(),
      "TPR": tpr.tolist()
    }


    return results

#
