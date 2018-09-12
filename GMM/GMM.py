# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 09:56:53 2017

@author: PXL4593
"""

import numpy as np
import pandas as pd
import random as rand
import matplotlib.pyplot as plt
from scipy.stats import norm


### Expectation-maximization Function

# probability that a point came from a Guassian with given parameters
def prob(val, mu, sig, lam):
  p = lam
  for i in range(len(val)):
    p *= norm.pdf(val[i], mu[i], sig[i][i])
  return p


# E step
def expectation(dataFrame, parameters):
  for i in range(dataFrame.shape[0]):
    x = dataFrame['x'][i]
    y = dataFrame['y'][i]
    # compute prob for (x,y) to be cluster 1
    p_cluster1 = prob([x, y], list(parameters['mu1']), list(parameters['sig1']), parameters['lambda'][0] )
    # compute prob for (x,y) to be cluster 1
    p_cluster2 = prob([x, y], list(parameters['mu2']), list(parameters['sig2']), parameters['lambda'][1] )
    # set cluster with higher probability
    if p_cluster1 > p_cluster2:
      dataFrame['label'][i] = 1
    else:
      dataFrame['label'][i] = 2
  return dataFrame


# M step
def maximization(dataFrame, parameters):
  points_assigned_to_cluster1 = dataFrame[dataFrame['label'] == 1]
  points_assigned_to_cluster2 = dataFrame[dataFrame['label'] == 2]
  # update lambda 
  percent_assigned_to_cluster1 = len(points_assigned_to_cluster1) / float(len(dataFrame))
  percent_assigned_to_cluster2 = 1 - percent_assigned_to_cluster1
  parameters['lambda'] = [percent_assigned_to_cluster1, percent_assigned_to_cluster2 ]
  # update mu
  parameters['mu1'] = [points_assigned_to_cluster1['x'].mean(), points_assigned_to_cluster1['y'].mean()]
  parameters['mu2'] = [points_assigned_to_cluster2['x'].mean(), points_assigned_to_cluster2['y'].mean()]
  # update sigma
  parameters['sig1'] = [[points_assigned_to_cluster1['x'].std(), 0 ], [ 0, points_assigned_to_cluster1['y'].std() ]]
  parameters['sig2'] = [[points_assigned_to_cluster2['x'].std(), 0 ], [ 0, points_assigned_to_cluster2['y'].std() ]]
  return parameters

# get the sum(distance between points)
def distance(old_params, new_params):
  dist = 0
  for param in ['mu1', 'mu2']:
    for i in range(len(old_params)):
      dist += (old_params[param][i] - new_params[param][i]) ** 2
  return dist ** 0.5



#######################################################################
#
# test
#
#######################################################################

# set random seed
rand.seed(123)

# 2 clusters
mu1 = [1, 5]
sig1 = [ [2, 0], [0, 3] ]

mu2 = [4, 0]
sig2 = [ [4, 0], [0, 1] ]

# generate samples
x1, y1 = np.random.multivariate_normal(mu1, sig1, 100).T
x2, y2 = np.random.multivariate_normal(mu2, sig2, 100).T

xs = np.concatenate((x1, x2))
ys = np.concatenate((y1, y2))
labels = ([1] * 100) + ([2] * 100)

data = {'x': xs, 'y': ys, 'label': labels}
df = pd.DataFrame(data=data)

# plot true figure
fig = plt.figure()
plt.scatter(df['x'], df['y'], 24, c=df['label'])

# loop until parameters converge
shift = 2**50
epsilon = 0.05
iters = 0
df_copy = df.copy()
# randomly assign points to their initial clusters
df_copy['label'] = np.random.choice(2, len(df))+1
# initialize the parameter
guess = { 'mu1': [np.mean(x1) + np.max(x1)*2*(np.random.random()-0.5),np.mean(x2)+ np.max(x2)*2*(np.random.random()-0.5)],
          'sig1': [ [np.std(x1), np.correlate(x1,x2)[0]*2*(np.random.random()-0.5)], [np.correlate(x1,x2)[0]*2*(np.random.random()-0.5), np.std(x2)] ],
          'mu2': [np.mean(x1)+np.max(x1)*2*(np.random.random()-0.5),np.mean(x2)+np.max(x2)*2*(np.random.random()-0.5)],
          'sig2': [ [np.std(x1), np.correlate(x1,x2)[0]*2*(np.random.random()-0.5)], [np.correlate(x1,x2)[0]*2*(np.random.random()-0.5), np.std(x2)] ],
          'lambda': [0.5, 0.5]
        }
params = pd.DataFrame(guess)


while shift > epsilon:
  iters += 1
  # E-step
  updated_labels = expectation(df_copy.copy(), params)

  # M-step
  updated_parameters = maximization(updated_labels, params.copy())

  # see if our estimates of mu have changed
  # could incorporate all params, or overall log-likelihood
  shift = distance(params, updated_parameters)

  # logging
  print("iteration {}, shift {}".format(iters, shift))

  # update labels and params for the next iteration
  df_copy = updated_labels
  params = updated_parameters

# plot estimate figure
fig = plt.figure()
plt.scatter(df_copy['x'], df_copy['y'], 24, c=df_copy['label'])