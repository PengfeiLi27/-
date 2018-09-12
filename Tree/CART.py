# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 16:35:46 2017

@author: PXL4593
"""
import math

class decisionnode:
    def __init__(self,col = -1,value = None, results = None, tb = None,fb = None):
        self.col = col   # col是待检验的判断条件所对应的列索引值
        self.value = value # value对应于为了使结果为True，当前列必须匹配的值
        self.results = results #保存的是针对当前分支的结果，它是一个字典
        self.tb = tb ## desision node,对应于结果为true时，树上相对于当前节点的子树上的节点
        self.fb = fb ## desision node,对应于结果为true时，树上相对于当前节点的子树上的节点

def dict_class(dataSet):
    classList = [data[-1] for data in dataSet]
    items = dict([(classList.count(i),i) for i in classList])
    return items
    
def computeEntropy(dataSet):
    """
    input: dataset
    output: entropy of dataset
    """
    numOfData = len(dataSet)
    classList = [data[-1] for data in dataSet]
    items = dict([(i,classList.count(i)) for i in classList])
    Entropy = 0.0
    for key in items:
        prob = float(items[key])/numOfData
        Entropy -= prob * math.log(prob, 2)
    return Entropy

