#! /bin/python3
'''
    Script to perform image segmentation and feature extraction in all images.
'''

#%%
# Imports
import os

# from my_tools import adjust_color
from my_tools import save_features_data
from my_tools import process_all

#%%
# Read the data
positions_file = "./position_data.csv"

img_path = "./images"
img_files = os.listdir(img_path)
img_files.sort()

out_path = "./output"
if not os.path.exists(out_path):
    os.mkdir(out_path)

#%%
# Process all images
# TODO: Can we do a parallel processing of this function?
# https://www.machinelearningplus.com/python/parallel-processing-python/
features_list, unknown_list = process_all(img_path, img_files, positions_file)
print('\n')
print("DONE!")

#%%
# Save extracted features
for i,file in enumerate(img_files):
    file_path = os.path.join(out_path, file[:-4]+'.csv')
    save_features_data(file_path, features_list[i])

