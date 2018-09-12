# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 11:50:08 2017

@author: PXL4593
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:05:33 2017

@author: PXL4593
"""
from math import log
import numpy as np
import treePlotter

def createDataSet(index = 1):
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true 
    """
    if index == 1:
        dataSet = [[0, 0, 0, 0, 'N'], 
                   [0, 0, 0, 1, 'N'], 
                   [1, 0, 0, 0, 'Y'], 
                   [2, 1, 0, 0, 'Y'], 
                   [2, 2, 1, 0, 'Y'], 
                   [2, 2, 1, 1, 'N'], 
                   [1, 2, 1, 1, 'Y']]
        labels = ['outlook', 'temperature', 'humidity', 'windy']
    
    elif index == 2:
        dataSet = [[0, 0, 0, 0, 'N'], 
                   [0, 0, 0, 1, 'N'], 
                   [1, 0, 0, 0, 'Y'], 
                   [2, 1, 0, 0, 'Y'], 
                   [2, 2, 1, 0, 'Y'], 
                   [2, 2, 1, 1, 'N'], 
                   [1, 2, 1, 1, 'Y'],
                   [0, 1, 0, 0, 'N'], 
                   [0, 2, 1, 0, 'Y'], 
                   [2, 1, 1, 0, 'Y'], 
                   [0, 1, 1, 1, 'Y'], 
                   [1, 1, 0, 1, 'Y'], 
                   [1, 0, 1, 0, 'Y'], 
                   [2, 1, 0, 1, 'N']]
        labels = ['outlook', 'temperature', 'humidity', 'windy']
    
    return dataSet, labels


def calculateEntropy(dataSet):
    """
    input: dataset
    output: entropy of dataset
    example: 
        [N,N,Y,Y,Y,N,Y]
        Entropy = -( 3/7 * log2(3/7) + 4/7 * log2(4/7)) = 0.9852
    """
    n_data = len(dataSet)
    dict_label = {}
    for i in range(n_data):
        if dataSet[i][-1] not in dict_label:
            dict_label[dataSet[i][-1]] = 1
        else:
            dict_label[dataSet[i][-1]] += 1
    
    Entropy = 0.0
    for i in dict_label.keys():
        prob = dict_label[i] / n_data
        Entropy -=  prob * log(prob,2)
        
    return Entropy
        

def majorityCnt(classList):
    """
    input: calss                                      e.g. [n,n,y,y,n,y,y] 
    output: the class that has majority count         e.g.  y
    """
    items = dict([(classList.count(i),i) for i in classList])
    return items[max(items.keys())]

def splitDataSet(dataSet, axis, value):
    """
    input: 
        dataSet = createDataSet(), 
        axis = 0
        value = 1                                    
    output: 
        [[0, 0, 0, 'Y'], [2, 1, 1, 'Y']]        
    """
    n_data = len(dataSet)
    output = []
    for i in range(n_data):
        tmp = dataSet[i]
        if tmp[axis] == value:
            tmp = tmp[:axis] + tmp[axis+1 :]
            output.append(tmp)
    return output

def chooseBestFeatureToSplit(dataSet):
    """
    input: dataSet
    output: best feature to split
    """
    n_feature = len(dataSet[0])-1
    n_data = float(len(dataSet))
    E = calculateEntropy(dataSet)
    dict_gain = {}
    d = np.asarray(dataSet)
    # axis
    for i in range(n_feature):
        # set of axis i
        set_axis = list(set(d[:,0]))
        sum_ = 0.0
        SplitInfo_ = 0.0
        for value in set_axis:
            v = int(float(value))
            subset = splitDataSet(dataSet, i, v)
            E_subset = calculateEntropy(subset)
            sum_ +=  E_subset * len(subset) / n_data
            if len(subset) == 0:
                SplitInfo_ -= 0
            else:
                SplitInfo_ -= len(subset) / n_data * log(len(subset) / n_data,2)
        
        dict_gain[i] = E - sum_
        if SplitInfo_ == 0:
            continue
        dict_gain[i] = dict_gain[i] / SplitInfo_
    # return the key with maximum value
    return max(dict_gain, key=dict_gain.get)

def createTree(dataSet, labels):
    """
    input： dataSet, labels
    output： decision tree
    method: recursion
    """
    classList = [example[-1] for example in dataSet]         
    # if only one label, return this label
    if classList.count(classList[0]) == len(classList):
        return classList[0]                                  
    
    # if no features, return majority of label
    if len(dataSet[0]) == 1:                                 
        return majorityCnt(classList)
    
    # compute best feature to split
    bestFeat = chooseBestFeatureToSplit(dataSet)  
           
    # name of best feature
    bestFeatLabel = labels[bestFeat]  
    
    # initalize tree
    myTree = {bestFeatLabel:{}}                   
    
    # delete best feature in label list
    del(labels[bestFeat])
    
    # featValues = [0, 0, 1, 2, 2, 2, 1]
    featValues = [example[bestFeat] for example in dataSet]   
    
    # set of [0, 0, 1, 2, 2, 2, 1] = [0,1,2]
    uniqueVals = set(featValues)
    
    # recursion
    for value in uniqueVals:
        subLabels = labels[:]                               
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree



dataSet, labels = createDataSet(1)
labels_tmp = labels[:]
desicionTree = createTree(dataSet, labels_tmp)
treePlotter.createPlot(desicionTree)