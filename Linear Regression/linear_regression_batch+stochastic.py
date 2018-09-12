# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:18:48 2017

@author: PXL4593
"""

import numpy as np
from sklearn.datasets.samples_generator import make_regression 
from sklearn import linear_model
import pylab

def gradient_descent(alpha, x, y, method = 'batch', lam = 0, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples
    
    # initial theta = [t0,t1]
    # model score: f(x) = t0 + t1 dot x
    t0 = np.random.random()
    t1 = np.random.random(x.shape[1])
    W = np.append(t0,t1)

    # cost function J(theta)
    J = sum([(t0 + np.dot(t1,x[i]) - y[i])**2 for i in range(m)])/2 + lam/2 * np.linalg.norm(W)**2
    
    if method == 'batch':
        while not converged:
            # calculate sum[ (h(x_i) - y_i) * x_i]
            # which is  sum[ (t0 + t1 dot x_i - y_i) * x_i]
            grad0 = 1.0/m * (sum([(t0 + np.dot(t1,x[i]) - y[i]) for i in range(m)]) + alpha * lam * t0)
            grad1 = 1.0/m * (sum([(t0 + np.dot(t1,x[i]) - y[i])*x[i] for i in range(m)]) + alpha * lam * t1)
    
            # update the theta
            t0 = t0 - (alpha * grad0 )
            t1 = t1 - (alpha * grad1 )
            W = np.append(t0,t1)
    
            # mean squared error = sum[ (h(x_i) - y_i)^2 ] 
            e = sum( [ (t0 + np.dot(t1,x[i]) - y[i])**2 for i in range(m)] )/2 + lam/2 * np.linalg.norm(W)**2
    
            if abs(J-e) <= ep:
                print ('Converged, iterations: ', iter, '!!!')
                converged = True
        
            J = e   # update error 
            iter += 1  # update iter
        
            if iter == max_iter:
                print ('Max interactions exceeded!')
                converged = True
                
    elif method == 'stochastic':
        while not converged:
            # calculate sum[ (h(x_i) - y_i) * x_i]
            # which is  sum[ (t0 + t1 dot x_i - y_i) * x_i]
            for i in range(m):
                grad0 = 1.0/m * (t0 + np.dot(t1,x[i]) - y[i])
                grad1 = 1.0/m * (t0 + np.dot(t1,x[i]) - y[i])*x[i]
    
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

    x, y = make_regression(n_samples=100, n_features=1, n_informative=1, 
                        random_state=7, noise=33) 
    
    
    alpha = 0.01 # learning rate
    ep = 0.01 # convergence criteria
    

    # call gredient decent
    theta0, theta1 = gradient_descent(alpha, x, y, 'batch', 0, ep, max_iter=10000)
    # theta0, theta1 = gradient_descent(alpha, x, y, 'batch',ep, max_iter=10000)
    print (('hand written: theta0 = %s theta1 = %s') %(theta0, theta1) )

    # check with sklearn linear regression 
    reg_sk = linear_model.LinearRegression()
    reg_sk.fit (x, y)
    print (('sklearn: theta0 = %s theta1 = %s') %(reg_sk.intercept_ , reg_sk.coef_) )
    
    # call gredient decent, with lambda = 5
    theta0_ridge, theta1_ridge = gradient_descent(0.0005 , x, y, 'batch', 5, ep, max_iter=10000)
    # theta0, theta1 = gradient_descent(alpha, x, y, 'batch',ep, max_iter=10000)
    print (('hand written: theta0_ridge = %s theta1_ridge = %s') %(theta0_ridge, theta1_ridge) )
    
    # check with sklearn RIDGE regression 
    reg_sk_ridge = linear_model.Ridge(alpha=5)
    reg_sk_ridge.fit (x, y)
    print (('Ridge_sklearn: theta0_ridge = %s theta1_ridge = %s') %(reg_sk_ridge.intercept_ , reg_sk_ridge.coef_) )
    
    
    # plot
    for i in range(x.shape[0]):
        y_predict = theta0_ridge + theta1_ridge * x 

    pylab.plot(x,y,'o')
    pylab.plot(x,y_predict,'k-')
    pylab.show()
    
    