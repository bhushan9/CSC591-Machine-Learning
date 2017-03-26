import pandas as pd
import numpy as np
import scipy
from sklearn.cluster import KMeans

# Read the csv file
df = pd.read_csv('MDP_Original_data.csv')
# Start with 6th column and iterate till last column 
#for i in range(6,len(df.columns)):
	#Use Kmeans to identify clusters

a= ['Level','symbolicRepresentationCount','probIndexinLevel','TotalTime', 'AppCount', 'hintCount', 'probIndexinLevel', 'WrongApp']

#	kmeans = KMeans(n_clusters=2, random_state=0).fit(df[df.columns[i]].values.reshape(-1,1))
	#Assign the labels to existing data column
#	df[df.columns[i]] = kmeans.labels_ 

new_df = df.loc[:,df.columns[0] : df.columns[5]]



for name in a:
	new_df.loc[:,name] = df[name]


#discretize the data (some features may need more than two levels)
for i in range(9,len(new_df.columns)):

	kmeans = KMeans(n_clusters=2, random_state=0).fit(new_df[new_df.columns[i]].values.reshape(-1,1))
	new=[x=x+1 for x in kmeans.labels_]
	new_df[new_df.columns[i]] = kmeans.labels_ 

print(df.columns.get_loc("reward"))





print(new_df.dtypes)

new_df.to_csv("test.csv", sep=',')
