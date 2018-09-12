    # -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:40:31 2017

@author: PXL4593
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn.metrics.pairwise as pair
from sklearn.metrics import mean_squared_error
from math import sqrt

# read data
header = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('ml-100k/u.data', sep='\t', names=header)

# compute number of id and number of film
n_users = np.shape(df.user_id.unique())[0]
n_items = np.shape(df.item_id.unique())[0]

# split data
train_data, test_data = train_test_split(df, test_size=0.25)

# create matrix
train_data_matrix = np.zeros((n_users, n_items))
for line in train_data.itertuples():
    train_data_matrix[line[1]-1, line[2]-1] = line[3]  
    
test_data_matrix = np.zeros((n_users, n_items))
for line in test_data.itertuples():
    test_data_matrix[line[1]-1, line[2]-1] = line[3]
    

# similarity
user_similarity = pair.cosine_similarity(train_data_matrix,train_data_matrix)
item_similarity = pair.cosine_similarity(train_data_matrix.T,train_data_matrix.T)

np.fill_diagonal(item_similarity,1.)
np.fill_diagonal(user_similarity,1.)

# small sample
train_smaple = np.array([[2,0,3],
                        [5,2,0],
                        [3,3,1],
                        [0,2,2]])

# similarity
user_sim_sample = pair.cosine_similarity(train_smaple,train_smaple)
item_sim_sample = pair.cosine_similarity(train_smaple.T,train_smaple.T)

np.fill_diagonal(item_sim_sample,1.)
np.fill_diagonal(user_sim_sample,1.)

def predict(ratings, similarity, type='user', meantype=1):
    if type == 'user':
        # average score that each user mark (include zero)
        mean_user_rating = ratings.mean(axis=1)
        
        if meantype == 2:
            # average score that each user mark (not include zero)
            tmp = (ratings!=0)
            mean_user_rating = ratings.sum(axis=1) / (tmp.sum(axis=1)+0.01)
        '''
        X = [1,2,3]
        X[:, np.newaxis] = [[1],
                            [2],
                            [3]]
        '''
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis]) 
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
    return pred

def rmse(prediction, ground_truth): 
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth)) 

item_prediction = predict(train_smaple, item_sim_sample , type='item')
user_prediction = predict(train_smaple, user_sim_sample, type='user', meantype=1)

'''
item_prediction = predict(train_data_matrix, item_similarity, type='item')
user_prediction = predict(train_data_matrix, user_similarity, type='user', meantype=2)

print ('User-based CF RMSE: ' + str(rmse(user_prediction, test_data_matrix)))
print ('Item-based CF RMSE: ' + str(rmse(item_prediction, test_data_matrix)))
'''
