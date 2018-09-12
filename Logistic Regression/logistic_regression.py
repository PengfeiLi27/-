# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:54:54 2017

@author: PXL4593
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:18:48 2017

@author: PXL4593
"""

import numpy as np
from math import exp
from sklearn import linear_model, datasets

def gradient_descent(alpha, x, y, method = 'batch', lam = 1,ep=0.0001, max_iter=1000000):
    converged = False
    iter = 0
    m = x.shape[0] # number of samples

    # initial theta = [t0,t1]
    # model score: f(x) = t0 + t1 dot x
    t0 = np.random.random()
    t1 = np.random.random(x.shape[1])
    
    #t0 = -0.5
    #t1 = np.array([2.,-4.])
    W = np.append(t0,t1)
   
    # cost function J(theta)
    J_1 = 1.0/m * -sum([(np.log(1/float(1+exp(-(t0 + np.dot(t1,x[i]))))) * y[i]) for i in range(m)])
    J_2 = 1.0/m * -sum([(np.log(1- 1/float(1+exp(-(t0 + np.dot(t1,x[i]))))) * (1-y[i])) for i in range(m)])
    J = J_1 + J_2 + lam/2 * np.linalg.norm(W)**2
    
    if method == 'batch':
        while not converged:
            # calculate sum[ (h(x_i) - y_i) * x_i]
            # which is  sum[ (t0 + t1 dot x_i - y_i) * x_i]
            grad0 = 1.0/m * (sum([(1/float(1+exp(-(t0 + np.dot(t1,x[i])))) - y[i]) for i in range(m)]) + alpha * lam * t0)         
            grad1 = 1.0/m * (sum([(1/float(1+exp(-(t0 + np.dot(t1,x[i])))) - y[i])*x[i] for i in range(m)]) + alpha * lam * t1)
            
            # update the theta
            t0 -= alpha * grad0
            t1 -= alpha * grad1
            W = np.append(t0,t1)
    
            # mean squared error = sum[ (h(x_i) - y_i)^2 ] 
            e_1 =  1.0/m * -sum([(np.log(1/float(1+exp(-(t0 + np.dot(t1,x[i]))))) * y[i]) for i in range(m)])
            e_2 =  1.0/m * -sum([(np.log(1- 1/float(1+exp(-(t0 + np.dot(t1,x[i]))))) * (1-y[i])) for i in range(m)])
            e = e_1 + e_2 + lam/2 * np.linalg.norm(W)**2
            
            if abs(J-e) <= ep:
                print ('Converged, iterations: ', iter, '!!!')
                converged = True
        
            J = e   # update error 
            #print (J)
            iter += 1  # update iter
        
            if iter == max_iter:
                print ('Max interactions exceeded!')
                converged = True
                
    elif method == 'stochastic':    
        while not converged:
            # calculate sum[ (h(x_i) - y_i) * x_i]
            # which is  sum[ (t0 + t1 dot x_i - y_i) * x_i]
            for i in range(m):
                grad0 = 1.0/m * (1/float(1+exp(-(t0 + np.dot(t1,x[i])))) - y[i])
                grad1 = 1.0/m * (1/float(1+exp(-(t0 + np.dot(t1,x[i])))) - y[i])*x[i]
            
                # update the theta
                t0 -= alpha * grad0
                t1 -= alpha * grad1
    
            # mean squared error = sum[ (h(x_i) - y_i)^2 ] 
            e_1 =  1.0/m * -sum([(np.log(1/float(1+exp(-(t0 + np.dot(t1,x[i]))))) * y[i]) for i in range(m)])
            e_2 =  1.0/m * -sum([(np.log(1- 1/float(1+exp(-(t0 + np.dot(t1,x[i]))))) * (1-y[i])) for i in range(m)])
            e = e_1 + e_2 
    
            if abs(J-e) <= ep:
                print ('Converged, iterations: ', iter, '!!!')
                converged = True
        
            J = e   # update error 
            #print (J)
            iter += 1  # update iter
        
            if iter == max_iter:
                print ('Max interactions exceeded!')
                converged = True
                
    else:
        print ("no such method")
        

    return t0,t1

if __name__ == '__main__':
    
    iris = datasets.load_iris()
    x = iris.data[0:100, :2]
    y = iris.target[0:100]
 
    alpha = 0.1 # learning rate
    ep = 0.00001 # convergence criteria

    # call gredient decent, and get intercept(=theta0) and slope(=theta1)
    theta0, theta1 = gradient_descent(alpha, x, y, 'batch', 1 ,ep, max_iter=10000)
    # theta0, theta1 = gradient_descent(alpha, x, y, 'batch',ep, max_iter=10000)
    print (('theta0 = %s theta1 = %s') %(theta0, theta1) )

    # check with sklearn logistic regression 
    logreg = linear_model.LogisticRegression(penalty="l2",C=1)
    logreg.fit(x, y)
    a = logreg.intercept_
    b = logreg.coef_
    print (('theta0 = %s theta1 = %s') %(a,b) )
    
    print('predict_train: ')
    predict = np.asarray([1/float(1+exp(-(theta0 + np.dot(theta1,x[i])))) for i in range(x.shape[0])])
    predict[predict>=0.5] =  1
    predict[predict<=0.5] =  0
    
    print(predict)