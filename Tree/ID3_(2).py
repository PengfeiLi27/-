# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 13:05:33 2017

@author: PXL4593
"""
from math import log
import numpy as np
import treePlotter
import matplotlib.pyplot as plt

def createDataSet():
    """
    outlook->  0: sunny | 1: overcast | 2: rain
    temperature-> 0: hot | 1: mild | 2: cool
    humidity-> 0: high | 1: normal
    windy-> 0: false | 1: true 
    """
    dataSet = [[0, 0, 0, 0, 'N'], 
               [0, 0, 0, 1, 'N'], 
               [1, 0, 0, 0, 'Y'], 
               [2, 1, 0, 0, 'Y'], 
               [2, 2, 1, 0, 'Y'], 
               [2, 2, 1, 1, 'N'], 
               [1, 2, 1, 1, 'Y']]
    labels = ['outlook', 'temperature', 'humidity', 'windy']
    return dataSet, labels


def calculateEntropy(dataSet):
    """
    input: dataset
    output: entropy of dataset
    """
    numOfData = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1      # 数每一类各多少个， {'Y': 4, 'N': 3}
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numOfData
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

def majorityCnt(classList):
    """
    input: calss                                      e.g. [n,n,y,y,n,y,y] 
    output: the class that has majority count         e.g.  y
    """
    items = dict([(classList.count(i),i) for i in classList])
    return items[max(items.keys())]

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:                      
            reduceFeatVec = featVec[:axis]              
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)            
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    """
    输入：数据集
    输出：最好的划分维度
    描述：选择最好的数据集划分维度
    """
    numFeatures = len(dataSet[0]) - 1                 #feature个数
    baseEntropy = calculateEntropy(dataSet)             #整个dataset的熵
    bestInfoGainRatio = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]  #每个feature的list
        uniqueVals = set(featList)                      #每个list的唯一值集合                 
        newEntropy = 0.0
        splitInfo = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)  #每个唯一值对应的剩余feature的组成子集
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calculateEntropy(subDataSet)
            splitInfo += -prob * log(prob, 2)
        infoGain = baseEntropy - newEntropy              #这个feature的infoGain
        if (splitInfo == 0): # fix the overflow bug
            continue
        infoGainRatio = infoGain / splitInfo             #这个feature的infoGainRatio      
        if (infoGainRatio > bestInfoGainRatio):          #选择最大的gain ratio
            bestInfoGainRatio = infoGainRatio
            bestFeature = i                              #选择最大的gain ratio对应的feature
    return bestFeature

def createTree(dataSet, labels):
    """
    输入：数据集，特征标签
    输出：决策树
    描述：递归构建决策树，利用上述的函数
    """
    classList = [example[-1] for example in dataSet]         # ['N', 'N', 'Y', 'Y', 'Y', 'N', 'Y']
    if classList.count(classList[0]) == len(classList):
        # classList所有元素都相等，即类别完全相同，停止划分
        return classList[0]                                  #splitDataSet(dataSet, 0, 0)此时全是N，返回N
    if len(dataSet[0]) == 1:                                 #[0, 0, 0, 0, 'N'] 
        # 遍历完所有特征时返回出现次数最多的
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)             #0－> 2   
        # 选择最大的gain ratio对应的feature
    bestFeatLabel = labels[bestFeat]                         #outlook -> windy     
    myTree = {bestFeatLabel:{}}                   
        #多重字典构建树{'outlook': {0: 'N'
    del(labels[bestFeat])                                    #['temperature', 'humidity', 'windy'] -> ['temperature', 'humidity']        
    featValues = [example[bestFeat] for example in dataSet]  #[0, 0, 1, 2, 2, 2, 1]     
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]                                #['temperature', 'humidity', 'windy']
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), subLabels)
            # 划分数据，为下一层计算准备
    return myTree



dataSet, labels = createDataSet()
labels_tmp = labels[:]
desicionTree = createTree(dataSet, labels_tmp)
treePlotter.createPlot(desicionTree)