# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:30:05 2017

@author: PXL4593
"""

import numpy as np
import pandas as pd

def list_to_array(obs,mapping):
    obs_seq = []
    for i in range(len(obs)):
        obs_seq.append(mapping[obs[i]])
        
    return np.asarray(obs_seq)
    
    

def Forward(trainsition_probability,emission_probability,theta,observation):
    # row: number of states
    # col: number of total observations
    row = trainsition_probability.shape[0]
    col = len(observation)
    
    F = np.zeros((row,col))          
    # t_1            
    F[:,0] = theta * emission_probability[:,observation[0]].T  
    # t_2 to t_col
    for t in range(1,col):              
        for n in range(row):                     
            F[n,t] = np.dot(F[:,t-1],trainsition_probability[:,n])*emission_probability[n,observation[t]]   

    return F

def Backward(trainsition_probability,emission_probability,theta,observation):
    # row: number of states
    # col: number of total observations
    row = trainsition_probability.shape[0]
    col = len(observation)
    
    B = np.zeros((row,col))  
    # t_1  
    B[:,(col-1):] = 1                  
    # t_2 to t_col
    for t in reversed(range(col-1)):
        for n in range(row):
            B[n,t] = np.sum(B[:,t+1]*trainsition_probability[n,:]*emission_probability[:,observation[t+1]])
            
    return B  

def Vertibi(trainsition_probability,emission_probability,theta,observation,hidden):
    # row: number of states
    # col: number of total observations
    row = trainsition_probability.shape[0]
    col = len(observation)
    
    V = np.zeros((row,col))          
    # t_1            
    V[:,0] = theta * emission_probability[:,observation[0]].T
    # t_2 to t_col
    
    for t in range(1,col):              
        for n in range(row):                     
            V[n,t] = max(V[:,t-1] * trainsition_probability[:,n])*emission_probability[n,observation[t]]   
    
    index = np.argmax(V,0).tolist()
    VV = []
    for i in range(col):
        VV.append(hidden[index[i]])
    return VV

def Baum_welch(A,B,pi,obs, criterion=1e-3):
    # observation map
    weatherObsMap   = {'no' : 0, 'yes' : 1}
    
    observations = list_to_array(obs,weatherObsMap)
    n_states = A.shape[0]
    n_samples = len(observations)

    done = False
    while not done:
        
        alpha = Forward(A,B,pi,observations)
        beta = Backward(A,B,pi,observations)
        
        xi = np.zeros((n_states,n_states,n_samples-1))
        for t in range(n_samples-1):
            denom = np.dot(np.dot(alpha[:,t].T, A) * B[:,observations[t+1]].T, beta[:,t+1])
            for i in range(n_states):
                numer = alpha[i,t] * A[i,:] * B[:,observations[t+1]].T * beta[:,t+1].T
                xi[i,:,t] = numer / denom

        
        gamma = np.sum(xi,axis=1)
     
        prod =  (alpha[:,n_samples-1] * beta[:,n_samples-1]).reshape((-1,1))
        gamma = np.hstack((gamma,  prod / np.sum(prod))) 
        
        # update A, B, pi
        newpi = gamma[:,0]
        newA = np.sum(xi,2) / np.sum(gamma[:,:-1],axis=1).reshape((-1,1))
        newB = np.copy(B)
        num_levels = B.shape[1]
        sumgamma = np.sum(gamma,axis=1)
        for lev in range(num_levels):
            mask = observations == lev
            newB[:,lev] = np.sum(gamma[:,mask],axis=1) / sumgamma
        
        # stop criteia
        if np.linalg.norm(pi - newpi) < criterion and np.linalg.norm(A - newA) < criterion and np.linalg.norm(B - newB) < criterion:
            done = 1

        A[:], B[:], pi[:] = newA, newB, newpi
        
    return A,B,pi

def Test(A,B,pi,obs,actual_state):
    # state map
    weatherStateMap   = {'sunny' : 0, 'rainy' : 1, 'foggy' : 2}
    
    # observation map
    weatherObsMap   = {'no' : 0, 'yes' : 1}
    
    # hidden state list
    hidden = list(weatherStateMap.keys())
    
    # convert string to category
    obs_seq = list_to_array(obs,weatherObsMap)
    
    # Vertibi
    predict_state = Vertibi(A,B,pi,obs_seq,hidden)
    
     # compute accurcy
    accuracy = sum([1 for i in range(len(predict_state)) if predict_state[i] == actual_state [i]]) / len(predict_state)
    
    return accuracy
    
def Test_(file,A,B,pi,actual_state):
    data = pd.read_csv(file, header = None)
    obs = data[1].tolist()
    actual_state = data[0].tolist()
    accuracy_test = Test(A,B,pi,obs,actual_state)
    
    return accuracy_test 

if __name__ == '__main__':
    
    # A
    trainsition_probability = np.array([[0.7,0.3],[0.4,0.6]])
    # B
    emission_probability = np.array([[0.1,0.4,0.5],[0.6,0.3,0.1]])
    # Theta
    start_probability = np.array([0.6,0.4])
    # Hidden state
    hidden = ['rainy','sunny']
    # observation (walk, shop, clean)
    observation = np.array([0,1,2])

    # compute P((walk,shop,shop)|Theta,A,B) by forward
    F = Forward(trainsition_probability,emission_probability,start_probability,observation)
    probability_F = sum(F,0)[::-1][0]
    
    print ("prob_forward = {}".format(probability_F))
    
    # compute P((walk,shop,shop)|Theta,A,B) by backward
    B = Backward(trainsition_probability,emission_probability,start_probability,observation)
    BB = start_probability*B[:,0]*emission_probability[:,observation[0]]    
    probability_B = sum(BB)
    
    print ("prob_backward = {}".format(probability_B))
    
    # Vertibi
    V = Vertibi(trainsition_probability,emission_probability,start_probability,observation,hidden)
    print ("with most probability, weather is {}".format(V))
    
    
    
    '''
    small sample
    known A, B, pi
    '''
    obs = ['no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes']
    actual_state = ['foggy', 'foggy', 'foggy', 'rainy', 'sunny', 'foggy', 'rainy', 'rainy', 'foggy', 'rainy']
    hidden = ['sunny', 'rainy', 'foggy']
    
    # prior parameter
    A = np.array([[ 0.7,  0.15,  0.15],
                  [ 0.2,  0.6,  0.2],
                  [ 0.2,  0.4,  0.4]])

    B = np.array([[ 0.95,  0.05],
                  [ 0.1,  0.9],
                  [ 0.95,  0.05]])

    pi = np.array([0.2, 0.4, 0.4])
    
    # accuracy
    accuracy_sample = Test(A,B,pi,obs,actual_state)
    print ("accuracy_sample = {}".format(accuracy_sample))
    
    '''
    large dataset
    known A, B, pi
    '''
    accuracy_test1 = Test_('test1.txt',A,B,pi,actual_state)
    print ("accuracy_test1 = {}".format(accuracy_test1))
    
    accuracy_test2 = Test_('test2.txt',A,B,pi,actual_state)
    print ("accuracy_test2 = {}".format(accuracy_test2))
    
    '''
    large dataset
    use baum_welch to learn A, B, pi
    '''
    # initial parameter
    # prior parameter
    A_ = np.array([[ 0.6,  0.2,  0.2],
                  [ 0.2,  0.6,  0.2],
                  [ 0.2,  0.4,  0.4]])

    B_ = np.array([[ 0.7,  0.3],
                  [ 0.1,  0.9],
                  [ 0.7,  0.3]])

    pi_ = np.array([0.2, 0.4, 0.4])
    
    # baum_welch
    A,B,pi = Baum_welch(A_, B_, pi_, obs, criterion=1e-3)
    
    accuracy_test1_BW = Test_('test1.txt',A,B,pi,actual_state)
    print ("accuracy_test1_BW = {}".format(accuracy_test1_BW))
    
    accuracy_test2_BW = Test_('test2.txt',A,B,pi,actual_state)
    print ("accuracy_test2_BW = {}".format(accuracy_test2_BW))
    