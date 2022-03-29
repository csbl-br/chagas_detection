#! /bin/python3
'''
    Script to select samples for the training/test dataset
'''
#%% ##########
# Imports
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os

nCPU = os.cpu_count() - 1
#%%
# Read the features
parasite_features_file = "../results/features_dataset.csv"
parasite_features = pd.read_csv(parasite_features_file)

# Remove oversegmented parasites
# Assuming that segmented regions with more than 500 pixels contains the parasite, but do not represent it.
parasite_features = parasite_features[parasite_features.area <= 500]

# Read features of unknown objects
unknown_features_file = "../results/features_dataset_unknown.csv"
unknown_features = pd.read_csv(unknown_features_file)

# Remove  Infs and NANs
unknown_features.replace([np.inf, -np.inf], np.nan, inplace=True)
unknown_features = unknown_features.dropna()

#%%
# Build datasets
X = parasite_features.loc[:, "area":"curvature_energy"]
y = parasite_features['structure']
X_parasite = X.to_numpy()
y_parasite = y.to_numpy()
# x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

x_unknown = unknown_features.loc[:, "area":'curvature_energy']
y_unknown = unknown_features['structure']

x_unknown = x_unknown.to_numpy()
y_unknown = y_unknown.to_numpy()

# %%
# Build cluster
clf = NearestNeighbors(n_neighbors = 1, n_jobs = nCPU).fit(X_parasite)

# Get the distances
distances, indices = clf.kneighbors(x_unknown, 1)

# %%
# Get the order of the distances
distances_order = np.argsort(distances.T[0])

# Get the indices of the unknown objects closer to the parasites
unknown_indices = distances_order[:len(y_parasite)]

# %%
# Get the nearest unknown objects
nearest_unknown_features = unknown_features.iloc[distances_order[:len(y_parasite)+(distances == 0).sum()],:]

# Remove repeated objects
nearest_unknown_features = nearest_unknown_features.iloc[(distances == 0).sum():, :]

# Save it for training
nearest_unknown_features.to_csv('../data/unknown_features.csv', index=False)
parasite_features.to_csv('../data/parasite_features.csv', index=False)
