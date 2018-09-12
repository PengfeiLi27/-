# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 17:43:03 2017

@author: PXL4593
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:09:25 2017

@author: PXL4593
"""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_circles

class SVM(object):

    def __init__(self, kernel='linear', rbf_gamma = 1, C = 1000, epsilon = 0.001):
        self.kernel = kernel
        self.epsilon = epsilon
        # larger gamma -> over fit
        # small gama -> under fit
        self.gamma = rbf_gamma
        # penalty C
        self.C = C
        

    def _init_parameters(self, X, Y):
        '''
        initalize parameter
        '''
        self.X = X
        self.Y = Y
        # bias 
        self.b = 0.0
        # dimension of feature
        self.n = len(X[0])
        # number of sample
        self.N = len(Y)
        # set all alpha = 0
        self.alpha = [0.0] * self.N
        # calculate error for each sample
        self.E = [self._E_(i) for i in range(self.N)]
        # max iteration
        self.Max_Interation = 50000


    def _satisfy_KKT(self, i):
        '''
        Satisfy KKT
        
        y_i * g(x_i) >=1 {x_i|a=0}
                      =1 {x_i|0<a<C}
                     <=1 {x_i|a=C}
                 
        '''
        yg = self.Y[i] * self._g_(i)
        
        if abs(self.alpha[i])<self.epsilon:
            return yg > 1 or yg == 1
        
        elif abs(self.alpha[i]-self.C)<self.epsilon:
            return yg < 1 or yg == 1
        
        else:
            return abs(yg-1) < self.epsilon

    def is_stop(self):
        for i in range(self.N):
            satisfy = self._satisfy_KKT(i)

            if not satisfy:
                return False
        return True

    def _select_two_parameters(self):
        '''
        select alpha_1, alpha_2 to implement SMO
        '''
        index_list = [i for i in range(self.N)]

        i1_list_1 = list(filter(lambda i: self.alpha[i] > 0 and self.alpha[i] < self.C, index_list))
        i1_list_2 = list(set(index_list) - set(i1_list_1))

        i1_list = i1_list_1
        i1_list.extend(i1_list_2)

        for i in i1_list:
            if self._satisfy_KKT(i):    
                continue

            E1 = self.E[i]
            max_ = (0, 0)

            for j in index_list:
                if i == j:
                    continue

                E2 = self.E[j]
                if abs(E1 - E2) > max_[0]:
                    max_ = (abs(E1 - E2), j)

            return i, max_[1]

    def _K_(self, x1, x2):
        '''
        kernel
        '''

        if self.kernel == 'linear':
            return sum([x1[k] * x2[k] for k in range(self.n)])
        if self.kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)])+1)**3
        if self.kernel == 'RBF':
            return np.exp(-self.gamma * sum([(x1[k] - x2[k])**2 for k in range(self.n)]))
                   

    def _g_(self, i):
        '''
        g(x_i) = sumj[a_j*y_j*K(x_j,x_i)]+b
        '''
        result = self.b

        for j in range(self.N):
            result += self.alpha[j] * self.Y[j] * self._K_(self.X[j], self.X[i])

        return result

    def _E_(self, i):
        '''
        E(i) = g(x_i) - y_i
        '''
        return self._g_(i) - self.Y[i]

   
    def train(self, features, labels):
        k = 0
        self._init_parameters(features, labels)
        
        while k < self.Max_Interation or self.is_stop():
            
            i1, i2 = self._select_two_parameters()
            
            if self.Y[i1] != self.Y[i2]:
                L = max(0, self.alpha[i2] - self.alpha[i1])
                H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])
            else:
                L = max(0, self.alpha[i2] + self.alpha[i1] - self.C)
                H = min(self.C, self.alpha[i2] + self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            
            '''
            eta = k11 + k22 - 2 k12
            '''
            eta = self._K_(self.X[i1], self.X[i1]) + self._K_(self.X[i2], self.X[i2]) - 2 * self._K_(self.X[i1], self.X[i2])     
            
            
            # 7.106
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta 

            # 7.108
            alph2_new = 0
            if alpha2_new_unc > H:
                alph2_new = H
            elif alpha2_new_unc < L:
                alph2_new = L
            else:
                alph2_new = alpha2_new_unc

            # 7.109
            alph1_new = self.alpha[i1] + self.Y[i1] * self.Y[i2] * (self.alpha[i2] - alph2_new)

            # 7.115 7.116
            b_new = 0
            b1_new = -E1 - self.Y[i1] * self._K_(self.X[i1], self.X[i1]) * (alph1_new - self.alpha[i1]) - self.Y[i2] * self._K_(self.X[i2], self.X[i1]) * (alph2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self._K_(self.X[i1], self.X[i2]) * (alph1_new - self.alpha[i1]) - self.Y[i2] * self._K_(self.X[i2], self.X[i2]) * (alph2_new - self.alpha[i2]) + self.b

            if alph1_new > 0 and alph1_new < self.C:
                b_new = b1_new
            elif alph2_new > 0 and alph2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            self.alpha[i1] = alph1_new
            self.alpha[i2] = alph2_new
            self.b = b_new

            self.E[i1] = self._E_(i1)
            self.E[i2] = self._E_(i2)
            
            k+= 1

    def help_predict(self,x_j):
        '''
        f(x) = sign(sum[a*y_i*K(x,x_i)]+b)
        '''
        f = self.b

        for i in range(self.N):
            f += self.alpha[i]*self.Y[i]*self._K_(x_j,self.X[i])

        if f > 0:
            return 1
        else:
            return -1

    def predict(self,X):
        results = []

        for x in X:
            results.append(self.help_predict(x))

        return results


def scatterplot(x,y,title=''):
    x = np.asarray(x)
    y = np.asarray(y)
    plt.scatter(x[y == 1, 0],
                x[y == 1, 1],
                c='b', marker='x',
                label='1')
    plt.scatter(x[y == -1, 0],
                x[y == -1, 1],
                c='r',
                marker='s',
                label='-1')
    
    plt.xlim([min(x[:,0]), max(x[:,0])])
    plt.ylim([min(x[:,1]), max(x[:,1])])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.title(title)
    plt.show()   
     
def generate_xor_data(N=100,seed=1):
    np.random.seed(seed)
    X = np.random.randn(N, 2)
    y = np.logical_xor(X[:, 0] > 0,X[:, 1] > 0)
    y = np.where(y, 1, -1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=seed)
    
    return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()

def generate_circle_data(N=100,seed=1):
    np.random.seed(seed)
    X, y = make_circles(N, factor=.1, noise=.1)
    y[y==0]=-1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    return X_train.tolist(), X_test.tolist(), y_train.tolist(), y_test.tolist()


if __name__ == "__main__":
    # set random seed
    seed = np.random.randint(1000)
    
    # create xor data
    #X_train, X_test, y_train, y_test = generate_xor_data(200,seed)
    
    # create circle data
    X_train, X_test, y_train, y_test = generate_circle_data(200,seed)
    
    svm = SVM(kernel='RBF',rbf_gamma=4, C=1000)
    svm.train(X_train, y_train)
    
    test_predict = svm.predict(X_test)
    
    accuracy = accuracy_score(y_test,test_predict)
    auc = roc_auc_score(y_test, test_predict)
    
    print ("accuracy", accuracy)
    print ("auc", auc)
    
    scatterplot(X_train,y_train,'train data')
    scatterplot(X_test,y_test,'test data')
    scatterplot(X_test,test_predict,'predict result')