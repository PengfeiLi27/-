# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 12:53:29 2017

@author: PXL4593
"""
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs

def Kmeans(data,K,maxIter=10000):
    def distance(vectorA,vectorB):
        return np.linalg.norm(vectorA-vectorB)
    
    def randomCenter(data,K):
        d = data.shape[1]
        Center = np.zeros((K,d))
        for i in range(d):
            dmin = np.min(data[:,0])
            dmax = np.max(data[:,0])
            Center[:,0] = dmin + (dmax - dmin) * np.random.rand(K)
        
        return Center
    
    def converged(centroids1, centroids2):
         return np.allclose(centroids1,centroids2)
    
    def single_Kmeans(data,K,maxIter):
        n = data.shape[0] 
        centroids = randomCenter(data,K)
        label = np.zeros(n,dtype=np.int) 
        Cost = np.zeros(n) 
        Converged = False
        iter_count = 0
        while ((not Converged) and iter_count<=maxIter):
            old_centroids = np.copy(centroids)
            for i in range(n):
                min_dist = np.inf
                for j in range(K):
                    dist = distance(data[i],centroids[j])
                    if dist < min_dist:
                        min_dist = dist
                        label[i] = j
                Cost[i] = distance(data[i],centroids[label[i]])**2
    
            # update centroid
            for m in range(K):
                centroids[m] = np.mean(data[label==m],axis=0)
            Converged = converged(old_centroids,centroids)    
            iter_count += 1
            
        return centroids, label, np.sum(Cost)
    
    best_Cost = np.inf
    best_centroids = None
    best_label = None
        
    for i in range(10):
        centroids, label, Cost = single_Kmeans(data,K,maxIter)
        if Cost < best_Cost:
            best_Cost = Cost
            best_centroids = centroids
            best_label = label
    
    return best_centroids,best_label
                    
            
    

if __name__=="__main__":
    # generate dataset ([X1, X2],y)
    X, y_true = make_blobs(n_samples=300, centers=4,
                           cluster_std=0.60, random_state=0)
    
    '''
    # use package in python
    kmeans = KMeans(n_clusters=4)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    centers = kmeans.cluster_centers_
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
    '''
    
    # Hand-written Kmeans
    center,label = Kmeans(X,4)
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=label, s=50, cmap='viridis')
    
    # true cluster
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis')
