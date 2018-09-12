# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:18:48 2017

@author: PXL4593
"""

import numpy as np
from sklearn.datasets.samples_generator import make_regression 
import pylab
from scipy import stats
from sklearn import linear_model

def gradient_descent(alpha, x, y, method = 'batch',ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    # initial theta = [t0,t1]
    # model score: f(x) = t0 + t1 dot x
    t0 = np.random.random(1)
    t1 = np.random.random(x.shape[1])

    # cost function J(theta)
    J = sum([(t0 + np.dot(t1,x[i]) - y[i])**2 for i in range(m)])
    
    if method == 'batch':
        # Iterate Loop of batch
        while not converged:
            # calculate sum[ (h(x_i) - y_i) * x_i]
            # which is  sum[ (t0 + t1 dot x_i - y_i) * x_i]
            grad0 = 1.0/m * sum([(t0 + np.dot(t1,x[i]) - y[i]) for i in range(m)]) 
            grad1 = 1.0/m * sum([(t0 + np.dot(t1,x[i]) - y[i])*x[i] for i in range(m)])
    
            # update the theta
            t0 -= alpha * grad0
            t1 -= alpha * grad1
    
            # mean squared error = sum[ (h(x_i) - y_i)^2 ] 
            e = sum( [ (t0 + np.dot(t1,x[i]) - y[i])**2 for i in range(m)] ) 
    
            if abs(J-e) <= ep:
                print ('Converged, iterations: ', iter, '!!!')
                converged = True
        
            J = e   # update error 
            iter += 1  # update iter
        
            if iter == max_iter:
                print ('Max interactions exceeded!')
                converged = True
                
    elif method == 'stochastic':
        # Iterate Loop of stochastic
        while not converged:
            # calculate sum[ (h(x_i) - y_i) * x_i]
            # which is  sum[ (t0 + t1 dot x_i - y_i) * x_i]
            grad0 = 1.0/m * sum([(t0 + np.dot(t1,x[i]) - y[i]) for i in range(m)]) 
            grad1 = 1.0/m * sum([(t0 + np.dot(t1,x[i]) - y[i])*x[i] for i in range(m)])
    
            # update the theta
            t0 -= alpha * grad0
            t1 -= alpha * grad1
    
            # mean squared error = sum[ (h(x_i) - y_i)^2 ] 
            e = sum( [ (t0 + np.dot(t1,x[i]) - y[i])**2 for i in range(m)] ) 
    
            if abs(J-e) <= ep:
                print ('Converged, iterations: ', iter, '!!!')
                converged = True
        
            J = e   # update error 
            iter += 1  # update iter
        
            if iter == max_iter:
                print ('Max interactions exceeded!')
                converged = True
                
    else:
        print ("no such method")
        

    return t0,t1

if __name__ == '__main__':

    x, y = make_regression(n_samples=100, n_features=2, n_informative=1, 
                        random_state=0, noise=35) 
    print ('x.shape = %s y.shape = %s' %(x.shape, y.shape))
 
    alpha = 0.01 # learning rate
    ep = 0.01 # convergence criteria

    # call gredient decent, and get intercept(=theta0) and slope(=theta1)
    theta0, theta1 = gradient_descent(alpha, x, y, 'bat',ep, max_iter=10000)
    print (('theta0 = %s theta1 = %s') %(theta0, theta1) )

    # check with sklearn linear regression 
    reg = linear_model.LinearRegression()
    reg.fit (x, y)
    print (('theta0 = %s theta1 = %s') %(reg.intercept_ , reg.coef_) )
    

    '''
    # plot
    for i in range(x.shape[0]):
        y_predict = theta0 + theta1*x 

    pylab.plot(x,y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()
    print ("Done!")
    '''