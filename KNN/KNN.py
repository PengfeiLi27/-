# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:09:25 2017

@author: PXL4593
"""

import numpy as np
import operator
class KNNClassifier():
    def __init__(self,k=3):
        self._k=k
        
    def distance_index(self,inputX,trainX):
        dataSetSize = np.shape(trainX)[0]
        
        diffMat = np.tile(inputX,(dataSetSize,1))-trainX
        
        sqDiffMat = diffMat**2
       
        distances = (sqDiffMat.sum(axis=1))**0.5
      
        sortedDistIndicies = np.argsort(distances)
        
        return sortedDistIndicies
    
    def help_classify(self,sample,trainX,trainY):
        
        sortedDistIndicies = self.distance_index(sample,trainX)
        classCount={}
        
        for i in range(self._k):
            label=trainY[sortedDistIndicies[i]]
            classCount[label]=classCount.get(label,0)+1
            
        sorteditem=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
       
        return sorteditem[0][0]
    
    
    def classify(self,trainX,trainY,inputX):
        # n=1: [x,x,x]
        # n!=: [[x,x,x],[x,x,x]]
        n = len(np.shape(inputX))
        result=[]
        if n == 1: 
            result.append(self.help_classify(inputX,trainX,trainY))
        else:
            for i in range(len(inputX)):
                result.append(self.help_classify(inputX[i],trainX,trainY))
        return result
    
if __name__=="__main__":
    X_train = [[1,1.1],
              [1,1],
              [0,0],
              [0,0.1]]
    
    y_train = ['A','A','B','B']
    
    X_test = [[0,0.1],[0,0]]
    
    model = KNNClassifier(k=3)
    
    result = model.classify(X_train, y_train, X_test)
    print (result)
    