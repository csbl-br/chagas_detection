#! /bin/python3
'''
    Script to perform T. cruzi detection in mobile phone images.
'''

#%%
# Imports
import os
import pandas as pd
import numpy as np
# import argparse 
from matplotlib import pyplot as plt
from skimage.io import imread #, imsave
from my_tools import adjust_color, adjust_size, feature_extractor, felzen_segmentation, get_predictions, load_model
from my_tools import remove_big_areas

# TODO: Implement argparser
# parser = argparse.ArgumentParser()
# parser.add_argument('--data', type=str, default='./images', help='Path to input file')
# parser.add_argument('--model', type=str, default='./models/rnd_clf.pickle', help='Path to trained model')
# parser.add_argument('--threshold', type=float, default=0.8, help='Prediction probability threshold')
# parser.add_argument('--save', type=str, default='./output', help='Path to save predictions')
# parser.add_argument('--no_preprocess',action='store_true', default=False, help='Do not pre process images')

#%%
# Read the data
img_path = "./images"
img_files = os.listdir(img_path)
img_files.sort()

out_path = "./output"
if not os.path.exists(out_path):
    os.mkdir(out_path)

#%%
# Process single image
file = img_files[2]
img = imread(os.path.join(img_path, file))
model = load_model("./models/rnd_clf.pickle")

# Adjust image size
img = adjust_size(img)
img_lab = adjust_color(img)

# Segmentation with Felzenszwalb
segments_fz = felzen_segmentation(img)

# Remove segments with area >= 3000 pixels
segments_fz = remove_big_areas(segments_fz)

#%%
# Get the features of each parasite in an image
''' TODO: Feature extraction method gives some warnings about feature calculation.
    This is generating NaNs and Infs to the features_data. This may cause some 
    errors while running the program in images that were preprocessed.
'''
features_data = feature_extractor(img_lab, segments_fz)

# Replace Infs and drop NaNs
features_data.replace([np.inf, -np.inf], np.nan, inplace=True)#.dropna()
features_data = features_data.dropna()

# Select the features
regions_features = features_data.loc[:, "area":"curvature_energy"]


#%%
# Get the predictions for each region
preds, probs = get_predictions(regions_features, model)
features_data["predictions"] = preds
features_data["probabilities"] = probs

#%%
# Select parasite positions
predicted_features = features_data[features_data.probabilities >= 0.8]
centroid_x = predicted_features["centroid_x"]
centroid_y = predicted_features["centroid_y"]

#%%
# Plot the image with detected parasites
plt.figure(figsize=(8,8))
plt.imshow(img)

for x,y in zip(centroid_x, centroid_y):
    circle = plt.Circle((x, y), 50, edgecolor="#0072b2", lw=2, fill=False)
    plt.text(x, y, "P", color="#0072b2", fontsize=10,
        verticalalignment='center', horizontalalignment='center')
    plt.gca().add_patch(circle)

plt.title(file)
plt.axis(False)
# plt.scatter(centroid_x, centroid_y, c='#0072b2')
plt.show()
# plt.savefig(os.path.join(out_path, "temp.png"))
