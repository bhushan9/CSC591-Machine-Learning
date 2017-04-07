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

test_df = df.loc[:,df.columns[0] : df.columns[5]]

out = np.loadtxt('ecr_2.txt',delimiter = ',')
max_ecr = np.amax(out, axis=0)
clusters = out.argmax(axis=0) + 2

# print max_ecr
# print clusters
# print np.amax(max_ecr)

# for i in [127,124,106,75,13,116,73,43,112,83,45,69,25,119,56,18,78,59,120,88,102,68,113,95,99,111,123,16,17,58,10]:
# 	print "(" + str(i) + "," + df.columns[i] + "," + str(clusters[i-6]) + ")"

for i,n_clusters in [(127,3),(18,2),(83,3),(78,3),(13,3),(25,3),(16,2),(17,2)]:
	column_name = df.columns[i]
	column = df[column_name]
	n_unique = len(np.unique(column))

	if n_unique > 6:
		# Reshape the data to make it multi-dimensional data
		kmeans = KMeans(n_clusters=n_clusters).fit(df[df.columns[i]].values.reshape(-1,1))
		test_df[column_name] = kmeans.labels_
	else:
		test_df[column_name] = df[column_name]

test_df.to_csv('test.csv', sep=',')