import pandas
import numpy as np
import scipy
from sklearn.cluster import KMeans
import os
from scipy.stats.stats import pearsonr
import collections
import mdptoolbox, mdptoolbox.example
import argparse
from copy import deepcopy

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
    original_data['state'] = original_data[features].apply(lambda x: ':'.join(str(v) for v in x), axis=1)
    # original_data['state'] = original_data[features].apply(tuple, axis=1)
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
    # distinct_states didn't contain terminal state
    student_list = list(data['student'].unique())
    distinct_states = list()
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        # don't consider last row
        temp_states = list(student_data['state'])[0:-1]
        distinct_states = distinct_states + temp_states
    distinct_states = list(set(distinct_states))

    Ns = len(distinct_states)

    # we include terminal state
    start_states = np.zeros(Ns + 1)
    A = np.zeros((Nx, Ns + 1, Ns + 1))
    expectR = np.zeros((Nx, Ns + 1, Ns + 1))

    # update table values episode by episode
    # each episode is a student data set
    for student in student_list:
        student_data = data.loc[data['student'] == student,]
        row_list = student_data.index.tolist()

        # count the number of start state
        start_states[distinct_states.index(student_data.loc[row_list[0], 'state'])] += 1

        # count the number of transition among states without terminal state
        for i in range(1, (len(row_list) - 1)):
            state1 = distinct_states.index(student_data.loc[row_list[i - 1], 'state'])
            state2 = distinct_states.index(student_data.loc[row_list[i], 'state'])
            act = student_data.loc[row_list[i], 'priorTutorAction']

            # count the state transition
            A[act, state1, state2] += 1
            expectR[act, state1, state2] += float(student_data.loc[row_list[i], 'reward'])

        # count the number of transition from state to terminal
        state1 = distinct_states.index(student_data.loc[row_list[-2], 'state'])
        act = student_data.loc[row_list[-1], 'priorTutorAction']
        A[act, state1, Ns] += 1
        expectR[act, state1, Ns] += float(student_data.loc[row_list[-1], 'reward'])

    # normalization
    start_states = start_states / np.sum(start_states)

    for act in range(Nx):
        A[act, Ns, Ns] = 1
        # generate expected reward
        with np.errstate(divide='ignore', invalid='ignore'):
            expectR[act] = np.divide(expectR[act], A[act])
            expectR[np.isnan(expectR)] = 0

        # each column will sum to 1 for each row, obtain the state transition table
        # some states only have either PS or WE transition to other state
        for l in np.where(np.sum(A[act], axis=1) == 0)[0]:
            A[act, l, l] = 1

        A[act] = np.divide(A[act].transpose(), np.sum(A[act], axis=1))
        A[act] = A[act].transpose()

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
df = pandas.read_csv('MDP_Original_data2.csv')

# Get the number of columns 
n = len(df.columns)

# Get corelation between columns of initial dataframe (NOT DISCRETIZED)
corelation_matrix = []
for i in range(6,n):
    row = []
    for j in range(6,n):
        if i==j:
            row.append(None)
        else:
            row.append(abs(pearsonr(df[df.columns[i]],df[df.columns[j]])[0]))
    corelation_matrix.append(row)

corelation_matrix = np.array(corelation_matrix) 
# Assign each feature to its own group 
group_list = {}
group = {}
for i in range(6,n):
    group_list[i] = [i]
    group[i] = i

# print np.amax(corelation_matrix)
# print corelation_matrix.argmax()
# x,y = np.unravel_index(corelation_matrix.argmax(),corelation_matrix.shape)
# print corelation_matrix[x,y]
# print corelation_matrix[y,x]

while np.amax(corelation_matrix) > 0.3:
    x,y = np.unravel_index(corelation_matrix.argmax(),corelation_matrix.shape)
    corelation_matrix[x][y] = None
    corelation_matrix[y][x] = None
    x = x + 6
    y = y + 6
    if group[x]!=group[y]:
        temp = group[y]
        group_list[group[x]] = group_list[group[x]] + group_list[group[y]]
        for each in group_list[group[y]]:
            group[each] = group[x]
        group_list.pop(temp)
        # group[y] = group[x]
print len(group_list.keys())

out = np.loadtxt('ecr_2.txt',delimiter = ',')
max_ecr = np.amax(out, axis=0)
clusters = out.argmax(axis=0) + 2

# Algorithm for selecting the best 8 features
# print np.amax(max_ecr)
# print max_ecr.argmax()

# Create a tuple (column_index, max_ecr, clusters)
ecr_data = []
for index,ecr in enumerate(max_ecr):
    ecr_data.append((index+6, ecr, clusters[index]))
ecr_data.sort(key = lambda x: x[1], reverse=True)

out = []
ecr_selected = []
index = 0
selected_groups = []
while (index<len(ecr_data) and len(ecr_selected)<8):
    candidate = ecr_data[index]
    candidate_group = group[index+6]
    if candidate_group not in selected_groups:
        selected_groups.append(candidate_group)
        ecr_selected.append(candidate)
        print "SELECTED : " + str(candidate) + " : " + str(candidate_group) + " : " + str(selected_groups)
    else:
        print "REJECTED : " + str(candidate) + " : " + str(candidate_group) + " : " + str(selected_groups)
    index += 1

print ecr_selected
test_df = df.loc[:,df.columns[0] : df.columns[5]]
for each in ecr_selected:
    i = each[0]
    column_name = df.columns[i]
    column = df[df.columns[i]]
    n_clusters = each[2]

    # Discretize the column if the number of unique elements is greater than 6
    n_unique = len(np.unique(column))
    # print n_unique

    if n_unique > 6:
      # Reshape the data to make it multi-dimensional data
      kmeans = KMeans(n_clusters=n_clusters).fit(df[df.columns[i]].values.reshape(-1,1))
      test_df[column_name] = kmeans.labels_
    else:
      test_df[column_name] = df[column_name]

test_df.to_csv('test.csv', sep=',')
# load data set with selected or extracted discrete features
[start_states, A, expectR, distinct_acts, distinct_states] = generate_MDP_input('test.csv')

# apply Value Iteration to run the MDP
vi = mdptoolbox.mdp.ValueIteration(A, expectR, discount = 0.9)
vi.run()

# output policy
# output_policy(distinct_acts, distinct_states, vi)

# evaluate policy using ECR
ecr = calcuate_ECR(start_states, vi.V)
print ecr

print 'Success!'