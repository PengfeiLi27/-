# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:35:05 2017

@author: PXL4593
"""

import pickle
import gzip
import numpy as np


def load_data():
    with gzip.open('mnist.pkl.gz','rb') as ff :
            u = pickle._Unpickler( ff )
            u.encoding = 'latin1'
            train, val, test = u.load()
    return train, val, test
    
def load_data_wrapper():
    tr_d, va_d, te_d = load_data()
    # Train
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))
    # Validation
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))
    # Test
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))
    return training_data, validation_data, test_data
    
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

