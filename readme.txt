#################
CORELATION METHOD
#################

1. python compute_ecr_for_each_feature.py 

Creates a multi-dimensional numpy array for ECR values of each feature at different discretization levels (2-6 clusters in KMeans). Creates output ecr_2.txt 

2. python discretize_data.py 

Reads ecr_2.txt, and discretizes data based on the best ECR value obtained. Stores it in MDP_Discretized_data2.csv

3. python filter_features_using_corelation.py 

Uses the discretized data, to run a co-relation algorithm over it, and select 8 input features for threshold values ranging from 0.08 to 0.40

4. python explore.py 

Get the best policy obtained from previous step, and run a greedy selection method over it. 
Observe values manually to select better alternative 


=================
CHECK BEST METHOD
=================
python MDP_process2.py -input final.csv

We stored the features from our best output to final.csv file


=====================
GROUP FEATURES METHOD
=====================
1. python compute_ecr_for_each_feature.py 

Creates a multi-dimensional numpy array for ECR values of each feature at different discretization levels (2-6 clusters in KMeans). Creates output ecr_2.txt 

2. python group_features_using_corelation.py 

Reads ecr_2.txt from previous step, and tries to group features into clusters using corelation values, and selects 8 features using the algorithm 

==================
FEATURE EXTRACTION
==================
Rscript feature_extract.R

Reads discretized data, and creates an extracted data file - MDP_Extracted_data2.csv. 
THE WORKING DIRECTORY MIGHT HAVE TO BE UPDATED BEFORE RUNNING THIS! 