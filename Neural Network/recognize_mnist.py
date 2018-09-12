# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:55:13 2017

@author: PXL4593
"""

import load_mnist as lm
from NN import Network

training_data,validation_data,test_data = lm.load_data_wrapper()

net = Network([784,30,10])

net.SGD(training_data,30,10,3.0,test_data = test_data)
