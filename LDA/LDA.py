# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 11:30:27 2017

@author: PXL4593
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def plot(X,title):
    with plt.style.context('seaborn-whitegrid'):
        plt.figure(figsize=(6, 4))
        for lab, col in zip((0, 1, 2),
                            ('blue', 'red', 'green')):
            plt.scatter(X[y==lab, 0],
                        X[y==lab, 1],
                        label=lab,
                        c=col)
        plt.xlabel('LDA 1')
        plt.ylabel('LDA 2')
        plt.legend(loc='lower center')
        plt.tight_layout()
        plt.title(title)
        plt.show()



df = pd.read_csv('iris.csv',header=None,sep=',')

df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'class']
df.dropna(how="all", inplace=True) # drops the empty line at file-end

x = df.iloc[:,0:4].values
y = df.iloc[:,4].values

# standardizing
X_std = StandardScaler().fit_transform(x)

# change class of y to 0,1,2
tmp = pd.get_dummies(y)
y = tmp.values.argmax(1)

# step 1: compute m1, m2 ,m3 where m1 = [m11,m12,m13,m14], 3 class, 4 feature
mean_vectors = []
for cl in range(0,3):
    mean_vectors.append(np.mean(x[y==cl], axis=0))

    
# step 2: compute with-in scatter matrix and between-class scatter matrix
S_W = np.zeros((4,4))
for cl,mv in zip(range(0,3), mean_vectors):
    class_sc_mat = np.zeros((4,4))                  # scatter matrix for every class
    for row in x[y == cl]:
        row, mv = row.reshape(4,1), mv.reshape(4,1) # make column vectors
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat                             # sum class scatter matrices



overall_mean = np.mean(x, axis=0)

S_B = np.zeros((4,4))
for i,mean_vec in enumerate(mean_vectors):  
    n = x[y==i,:].shape[0]
    mean_vec = mean_vec.reshape(4,1) # make column vector
    overall_mean = overall_mean.reshape(4,1) # make column vector
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

# step 3: solve the generalized eigenvalue problem for matrix inv(S_W)*(S_B)
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

for i in range(len(eig_vals)):
    eigvec_sc = eig_vecs[:,i].reshape(4,1)   

    
    
# step 4 calculate transform matrix W to convert 4d to 2d
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)   
    
W = np.hstack((eig_pairs[0][1].reshape(4,1), eig_pairs[1][1].reshape(4,1)))

# step 5 compute X_lda
X_lda = x.dot(W)

# For comparison
LDA = LinearDiscriminantAnalysis(n_components=2, solver='eigen')
LDA.fit(X_std,y)
X_lda_sk = LDA.transform(X_std)
X_lda_sk[:] = X_lda_sk[:]*-1

pca = PCA(n_components=2)
pca.fit(X_std)
X_PCA = pca.transform(X_std)
X_PCA[:,1] = X_PCA[:,1]*-1


# plot comparision
plot(X_lda,'LDA_hand')
plot(X_lda_sk,'LDA_sklearn')
plot(X_PCA,'PCA')

# 1 lda and lda_sk are same
# 2 different from pca
# 3 lda has better discrimincation 