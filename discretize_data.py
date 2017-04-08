import pandas
import numpy as np
import scipy
from sklearn.cluster import KMeans
import os
from scipy.stats.stats import pearsonr
import collections
import mdptoolbox, mdptoolbox.example
import argparse

# Read the csv file
df = pandas.read_csv('MDP_Original_data2.csv')

# Get the number of columns 
n = len(df.columns)

out = np.loadtxt('ecr_2.txt',delimiter = ',')
max_ecr = np.amax(out, axis=0)
clusters = out.argmax(axis=0) + 2

for index,ecr in enumerate(max_ecr):
    i = index + 6
    n_clusters = clusters[index]
    column_name = df.columns[i]
    column = df[column_name]
    n_unique = len(np.unique(column))

    if n_unique > 6:
        # Reshape the data to make it multi-dimensional data
        kmeans = KMeans(n_clusters=n_clusters).fit(df[column_name].values.reshape(-1,1))
        df[column_name] = kmeans.labels_
    else:
        df[column_name] = df[column_name]

df.to_csv('MDP_Discretized_data2.csv', sep=',', index=False)