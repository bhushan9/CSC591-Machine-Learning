import pandas
import numpy as np
import scipy
from sklearn.cluster import KMeans
import os
from scipy.stats.stats import pearsonr
import collections
import mdptoolbox, mdptoolbox.example
import argparse

# load data set with selected or extracted features, features are discrete
# features are the columns after reward column
def generate_MDP_input(filename):
    original_data = pandas.read_csv(filename)
    feature_name = list(original_data)
    reward_idx = feature_name.index('reward')
    start_Fidx = reward_idx + 1

    students_variables = ['student', 'priorTutorAction', 'reward']
    features = feature_name[start_Fidx: len(feature_name)]

    # generate distinct state based on feature
    original_data['state'] = original_data[features].apply(lambda x : ':'.join(str(v) for v in x), axis=1)
    students_variables = students_variables + ['state']
    data = original_data[students_variables]

    # quantify actions
    distinct_acts = list(data['priorTutorAction'].unique())
    Nx = len(distinct_acts)
    i = 0
    for act in distinct_acts:
        data.loc[data['priorTutorAction'] == act, 'priorTutorAction'] = i
        i += 1

    # initialize state transition table, expected reward table, starting state table
    distinct_states = list(data['state'].unique())
    Ns = len(distinct_states)
    start_states = np.zeros(Ns)
    A = np.zeros((Nx, Ns, Ns))
    expectR = np.zeros((Nx, Ns, Ns))

    # update table values episode by episode
    # each episode is a student data set
    student_list = list(data['student'].unique())
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()

        # count the number of start state
        start_states[distinct_states.index(student_data.loc[row_list[0], 'state'])] += 1

        for i in range(1, len(row_list)):
            state1 = distinct_states.index(student_data.loc[row_list[i-1], 'state'])
            state2 = distinct_states.index(student_data.loc[row_list[i], 'state'])
            act = student_data.loc[row_list[i], 'priorTutorAction']

            # count the state transition
            A[act, state1, state2] += 1
            expectR[act, state1, state2] += float(student_data.loc[row_list[i], 'reward'])

    # normalization
    start_states = start_states/np.sum(start_states)

    print Ns
    print start_states
    print A
    print expectR

    for act in range(Nx):
        # generate expected reward
        # it has the warning, ignore it
        with np.errstate(divide='ignore', invalid='ignore'):
            expectR[act] = np.divide(expectR[act], A[act])
            expectR[np.isnan(expectR)] = 0

        # each column will sum to 1 for each row, obtain the state transition table
        A[act] = np.divide(A[act].transpose(), np.sum(A[act], axis=1))
        A[act] = A[act].transpose()
        A[np.isnan(A)] = float(1)/Ns

    print Ns
    print start_states
    print A
    print expectR
    
    return [start_states, A, expectR, distinct_acts, distinct_states]

def calcuate_ECR(start_states, expectV):
        ECR_value = start_states.dot(np.array(expectV))
        return ECR_value

def output_policy(distinct_acts, distinct_states, vi):
    Ns = len(distinct_states)
    print('Policy: ')
    print('state -> action, value-function')
    for s in range(Ns):
        print(distinct_states[s]+ " -> " + distinct_acts[vi.policy[s]] + ", "+str(vi.V[s]))

# Read the csv file
df = pandas.read_csv('MDP_Original_data.csv')
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
for i in range(7,8):
	test_df = df.loc[:,df.columns[0] : df.columns[5]]
	column_name = discretized_df.columns[i]
	test_df[column_name] = discretized_df[column_name]
	test_df.to_csv('test.csv', sep=',')

	# load data set with selected or extracted discrete features
	[start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input('test.csv')

	# apply Value Iteration to run the MDP
	vi = mdptoolbox.mdp.ValueIteration(A, expectR, 0.9)
	vi.run()

	# output policy
	# output_policy(distinct_acts, distinct_states, vi)

	# evaluate policy using ECR
	ecr = calcuate_ECR(start_states, vi.V)
    
	# try:
	# 	ecr = float(os.popen('python MDP_process.py -input test.csv').read().split('\n')[-2].split(': ')[1])
	# except:
	# 	ecr = None
	out.append(ecr)

out = np.array(out)
print out

print 'Success!'