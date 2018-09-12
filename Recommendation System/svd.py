# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 17:23:25 2017

@author: PXL4593
"""
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# Utility matrix
M = np.array([ [ 5.,  5.,  3.,  0.,  5.,  5.],
               [ 5.,  0.,  4.,  0.,  4.,  4.],
               [ 0.,  3.,  0.,  5.,  4.,  5.],
               [ 5.,  4.,  3.,  3.,  5.,  5.]])

# compute svd, k = 2 means reduce dimension to 2-d
# u: user
# vt: item
# s: eigenvalue
u, s, vt = svds(M, k = 2)
s = np.diag(s)
label_user = np.array(['A','B','C','D'])
label_item = np.array(['s1','s2','s3','s4','s5','s6'])

# new user
New = np.array([5,5,0,0,0,5])

# dimension reduction of new user
New_2d = np.dot(np.dot(New,vt.T),np.linalg.inv(s))

# visualization
plt.scatter(u[:, 0],u[:, 1],c='b', marker='x')

plt.scatter(vt[0,:],vt[1,:],c='r', marker='o')

for i in range(M.shape[0]):
    plt.text(u[i,0]+1e-2, u[i,1]+1e-2, label_user[i])
    
for i in range(M.shape[1]):    
    plt.text(vt[0,i]+1e-2, vt[1,i]+1e-2, label_item[i])
     
plt.scatter(New_2d[0],New_2d[1],c='green', marker='*')
plt.text(New_2d[0]+1e-2, New_2d[1]+1e-2, 'New User')
      
plt.show()   

# find the most similar user by using cosine similarity
similar = [] 
for i in range(u.shape[0]):
    sim = np.dot(u[i],New_2d) / ( np.linalg.norm(u[i]) * np.linalg.norm(New_2d))
    similar.append(sim)

index = np.argsort(similar)[::-1][0]
print ("most similar user = {}".format(label_user[index]))

# recommend the film liked by most similar user 
recommend = {}
for i in range(len(New)):
    if M[index,i]!=0 and New[i]==0:
        recommend.update({label_item[i]:M[index,i]})

recommend_sort = sorted(recommend, key=recommend.get, reverse=True)      

print ("film recommend order is {}".format(recommend_sort))