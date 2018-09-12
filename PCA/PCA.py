# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 16:29:32 2017

@author: PXL4593
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

df = pd.read_csv(
    'iris.csv',
    header=None,
    sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

x = df.iloc[:,0:4].values
y = df.iloc[:,4].values

# standardizing
X_std = StandardScaler().fit_transform(x)

# step 1: calculate covariance matrix
# cov = 1/(n-1) * [ (X-Xbar)' dot (X-Xbar) ] 
# dim(cov) = d * d   where d is # of features
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
# calculate by np.cov
cov_mat_python = np.cov(X_std.T)

# step 2: perform an eigendecomposition on covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# 3 sort eigenpairs by eigenvalues
# percent of
eig_pairs = list(zip(eig_vals, eig_vecs.T))
eig_pairs.sort(key=lambda x: x[0], reverse=True)

# 4 explain variance
tot = sum(eig_vals)
var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

# plot
with plt.style.context('seaborn-whitegrid'):
    plt.figure(figsize=(6, 4))

    plt.bar(range(4), var_exp, alpha=0.5, align='center',
            label='individual explained variance')
    plt.step(range(4), cum_var_exp, where='mid',
             label='cumulative explained variance')
    plt.ylabel('Explained variance ratio')
    plt.xlabel('Principal components')
    plt.legend(loc='best')
    plt.tight_layout()
    
    
# 5 projection: covert 4-d to 2-d
# x_pca = x' * W
matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
                      eig_pairs[1][1].reshape(4,1)))    


X_PCA = X_std.dot(matrix_w)

# compute by sklearn
pca = PCA(n_components=2)
pca.fit(X_std)
X_PCA_sk = pca.transform(X_std)
X_PCA_sk[:,1] = X_PCA_sk[:,1]*-1

# pca.components_ is eigenvector
v1 = pca.components_
v2 = eig_vecs

# plot
def plot(X_PCA):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                            ('blue', 'red', 'green')):
            plt.scatter(X_PCA[y==lab, 0],
                        X_PCA[y==lab, 1],
                        label=lab,
                        c=col)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.legend(loc='lower center')
        plt.tight_layout()
        plt.show()
        
    
plot(X_PCA)
plot(X_PCA_sk)
    
# we can find that they are the same  
    
    
    
    
    
    
    
    
    
    
    
    