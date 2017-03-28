import pandas as pd
import numpy as np
import scipy
from sklearn.cluster import KMeans
import os
from copy import deepcopy

# Read the csv file
df = pd.read_csv('MDP_Original_data.csv')
# Start with 6th column and iterate till last column 
#for i in range(6,len(df.columns)):
	#Use Kmeans to identify cluster

a= ['Level','NewLevel','SolvedPSInLevel','probDiff','TotalTime','avgstepTimePS','RightApp']

#	kmeans = KMeans(n_clusters=2, random_state=0).fit(df[df.columns[i]].values.reshape(-1,1))
	#Assign the labels to existing data column
#	df[df.columns[i]] = kmeans.labels_ 

new_df = df.loc[:,df.columns[0] : df.columns[5]]

#store_df = new_df

# for name in a:
# 	new_df.loc[:,name] = df[name]



#new_df.loc[:,'f1']=4*df['cumul_difficultProblemCountSolved']+2*df['cumul_difficultProblemCountWE']+2*df['cumul_easyProblemCountSolved']+df['cumul_easyProblemCountWE']
#new_df.loc[:,'f2'] = df['avgstepTime'] - df['avgstepTimePS']

#discretize the data (some features may need more than two levels)

values_ecr = {} # stores max ecr value for a column
name = []
for i in range(7,10):
	print(df.columns[i])

	#test_df = deepcopy()
	# test_df = deepcopy(new_df)
	test_df = deepcopy(new_df)
	#store_df = test_df
	unique = len(np.unique(df[df.columns[i]]))
	if unique >5 :
		max_ecr = None
		n = None
		for no_of_clusters in range(2,6):
			kmeans = KMeans(n_clusters=no_of_clusters, random_state=0).fit(df[df.columns[i]].values.reshape(-1,1))
			test_df[df.columns[i]] = [x+1 for x in kmeans.labels_]
			#name.append(len(test_df.columns))
			# Write to csv file 
			test_df.to_csv("test.csv", sep=',')
			#print str('yolo') + str(test_df.columns)
			cmd = "python MDP_process.py -input test.csv"
			try:
				ecr = float(os.popen(cmd).read().split('\n')[-2].split(': ')[1])
			except:
				ecr = None
			if ecr is not None and (max_ecr is None or ecr>max_ecr):
				max_ecr = ecr
				n = no_of_clusters
				test_df[df.columns[i]]=[x+1 for x in kmeans.labels_]
				
		#values.append( str(df.columns[i]) + " : " + str(n) + " : " + str(max_ecr))
		values_ecr[df.columns[i]]=max_ecr
	else:
		test_df[df.columns[i]]=df[df.columns[i]]
		

test_df.to_csv("new.csv",sep=',')

print(values_ecr)
print(name)
# write optimal discretized data to file 


# print os.popen(cmd).read().split('\n')[-1]
# print float(os.popen(cmd).read().split('\n')[-2].split(': ')[1])
		# print str(df.columns[i]) + " : Discretize Bitch! : " + str(unique)
	# else:
	# 	print str(df.columns[i]) + " : Am I not perfect?"
		


		 

	





#calculate the chi square test




# new_df.to_csv("test.csv", sep=',')
