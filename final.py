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
df = pandas.read_csv('MDP_Discretized_data2.csv')

test_df = df.loc[:,df.columns[0] : df.columns[5]]

# out = np.loadtxt('ecr_2.txt',delimiter = ',')
# max_ecr = np.amax(out, axis=0)
# clusters = out.argmax(axis=0) + 2

# print max_ecr
# print clusters
# print np.amax(max_ecr)

# data = []
for i in [17, 126, 113, 18, 80, 106, 123, 59]:
	# print "(" + str(i) + "," + str(clusters[i-6]) + " : " + df.columns[i] + ")"
	# data.append((i,clusters[i-6]))
	column_name = df.columns[i]
	test_df[column_name] = df[column_name]
	print column_name
	

# print df.columns[1]
# print df[df.columns[1]]

# print data
# for i,n_clusters in data:
# 	column_name = df.columns[i]
# 	column = df[column_name]
# 	n_unique = len(np.unique(column))

# 	if n_unique > 6:
# 		# Reshape the data to make it multi-dimensional data
# 		kmeans = KMeans(n_clusters=n_clusters).fit(df[df.columns[i]].values.reshape(-1,1))
# 		test_df[column_name] = kmeans.labels_
# 	else:
# 		test_df[column_name] = df[column_name]

test_df.to_csv('test2.csv', sep=',', index=False)