import pandas as pd
import numpy as np
import scipy
from sklearn.cluster import KMeans

# Read the csv file
df = pd.read_csv('MDP_Original_data.csv')
# Start with 6th column and iterate till last column 
for i in range(6,len(df.columns)):
	#Use Kmeans to identify clusters
	kmeans = KMeans(n_clusters=2, random_state=0).fit(df[df.columns[i]].values.reshape(-1,1))
	#Assign the labels to existing data column
	df[df.columns[i]] = kmeans.labels_ 
	

df.to_csv("discretized_output.csv", sep='\t', encoding='utf-8')
