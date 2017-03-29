import pandas as pd
import numpy as np
import scipy
from sklearn.cluster import KMeans
import os
from scipy.stats.stats import pearsonr

# Read the csv file
df = pd.read_csv('MDP_Original_data.csv')
discretized_df = df.loc[:,df.columns[0] : df.columns[5]]
# print discretized_df

# Get the number of columns 
n = len(df.columns)

# Get corelation between columns of initial dataframe (NOT DISCRETIZED)
corelation_matrix = []
for i in range(6,n):
	row = []
	for j in range(6,n):
		row.append(pearsonr(df[df.columns[i]],df[df.columns[j]])[0])
	corelation_matrix.append(row)

corelation_matrix = np.array(corelation_matrix)
# print corelation_matrix

# The first six columns are static. Iterate over rest
for i in range(6,n):

	# # Get column name 
	column_name = df.columns[i]
	# print column_name

	# Get the column as a vector
	column = df[df.columns[i]]
	# print column

	# Discretize the column if the number of unique elements is greater than 6
	n_unique = len(np.unique(column))
	# print n_unique

	if n_unique > 6:
		# Reshape the data to make it multi-dimensional data
		kmeans = KMeans(n_clusters=3).fit(df[df.columns[i]].values.reshape(-1,1))
		discretized_df[column_name] = kmeans.labels_
	else:
		discretized_df[column_name] = df[column_name]

# print discretized_df

# Find ECR value for each feature independently
out = []
for i in range(6,15):
	test_df = df.loc[:,df.columns[0] : df.columns[5]]
	column_name = discretized_df.columns[i]
	test_df[column_name] = discretized_df[column_name]
	test_df.to_csv('test.csv', sep=',')

	try:
		ecr = float(os.popen('python MDP_process.py -input test.csv').read().split('\n')[-2].split(': ')[1])
	except:
		ecr = None
	out.append(ecr)

out = np.array(out)
print out

print 'Success!'