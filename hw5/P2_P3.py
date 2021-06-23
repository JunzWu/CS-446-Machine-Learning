# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 19:08:55 2019

@author: Junz
"""
import numpy as np
import hw5_utils
import hw5
import math

#Problem 2
'''
#Problem c
X, tex, Y, tey=hw5_utils.load_iris_data(ratio=1)
score = []
for k in range(2,21):
    C = hw5.k_means(X, k)
    score += [hw5.get_purity_score(X, Y, C)]
hw5_utils.line_plot(score)
'''
'''
#Problem f
X, tex, Y, tey=hw5_utils.load_iris_data(ratio=0.8)
train_error1 = []
train_error3 = []
test_error1 = []
test_error3 = []

for k in range(2,21):
    lr, C = hw5.classify_using_k_means(X, Y, k, l=1)
    n = X.shape[0]
    A = np.zeros((n,k))
    for i in range(n):
        distance = []
        for j in range(k):
            distance += [np.linalg.norm(X[i,:]-C[j,:])]
        j = distance.index(min(distance))
        A[i,j]=1
    p = lr.predict_proba(A)
    e = 0
    for i in range(p.shape[0]):
        a = np.argmax(p[i,:])
        if a != Y[i]:
            e += 1
    train_error1 += [e/Y.shape[0]]
for k in range(2,21):
    lr, C = hw5.classify_using_k_means(X, Y, k, l=1)
    n = tex.shape[0]
    A = np.zeros((n,k))
    for i in range(n):
        distance = []
        for j in range(k):
            distance += [np.linalg.norm(tex[i,:]-C[j,:])]
        j = distance.index(min(distance))
        A[i,j]=1
    p = lr.predict_proba(A)
    e = 0
    for i in range(p.shape[0]):
        a = np.argmax(p[i,:])
        if a != tey[i]:
            e += 1
    test_error1 += [e/tey.shape[0]]
for k in range(3,21):
    lr, C = hw5.classify_using_k_means(X, Y, k, l=3)
    n = X.shape[0]
    A = np.zeros((n,k))
    for i in range(n):
        distance1 = np.zeros(k)
        distance2 = np.zeros(k)
        for j in range(k):
            distance1[j] = np.linalg.norm(X[i,:]-C[j,:])
            distance2[j] = np.linalg.norm(X[i,:]-C[j,:])
        distance2 = distance2[np.argsort(distance2)]
        
        for a in range(3):
            j = np.where(distance1 == distance2[a])[0]
            for b in range(j.shape[0]):
                if A[i,j[b]] == 0:
                    A[i,j[b]] = 1
                    break
    p = lr.predict_proba(A)
    e = 0
    for i in range(p.shape[0]):
        a = np.argmax(p[i,:])
        if a != Y[i]:
            e += 1
    train_error3 += [e/Y.shape[0]]
for k in range(3,21):
    lr, C = hw5.classify_using_k_means(X, Y, k, l=1)
    n = tex.shape[0]
    A = np.zeros((n,k))
    for i in range(n):
        distance1 = np.zeros(k)
        distance2 = np.zeros(k)
        for j in range(k):
            distance1[j] = np.linalg.norm(tex[i,:]-C[j,:])
            distance2[j] = np.linalg.norm(tex[i,:]-C[j,:])
        distance2 = distance2[np.argsort(distance2)]
        
        for a in range(3):
            j = np.where(distance1 == distance2[a])[0]
            for b in range(j.shape[0]):
                if A[i,j[b]] == 0:
                    A[i,j[b]] = 1
                    break
    p = lr.predict_proba(A)
    e = 0
    for i in range(p.shape[0]):
        a = np.argmax(p[i,:])
        if a != tey[i]:
            e += 1
    test_error3 += [e/tey.shape[0]]


print(train_error1)
print(test_error1)
print(train_error3)
print(test_error3)
hw5_utils.line_plot(train_error1, test_error1)
hw5_utils.line_plot(train_error3, test_error3, min_k=3)
'''
'''
#Problem g
X, tex, Y, tey=hw5_utils.load_iris_data(ratio=1)
C = hw5.k_means(X, 4)
n = X.shape[0]
A = np.zeros((n,4))
for i in range(n):
    distance = []
    for j in range(4):
        distance += [np.linalg.norm(X[i,:]-C[j,:])]
    j = distance.index(min(distance))
    A[i,j]=1

A = A.astype(np.int32)
print(A)
j = []
for i in range(4):
    j += [np.where(A[:,i]==1)[0]]
hw5_utils.scatter_plot_2d_project(X[j[0],:], X[j[1],:], X[j[2],:], X[j[3],:], C)
'''

#Problem 3
#Problem b

X, tex, Y, tey=hw5_utils.load_iris_data(ratio=1)
score = []
n = X.shape[0]
d = X.shape[1]
for k in range(10, 11):
  while True:
    singular = False
    mu, covars, pai, R = hw5.gmm(X, k)
    for i in range(k):
        for j in range(d):
            if covars[i,j] == 0:
                singular = True
    if singular == True:
        continue
    
    for i in range(n):
        for j in range(k):
            if R[i,j] == 0:
                R[i,j] = 1.0e-300

    log = 0
    for i in range(n):
            for j in range(k):
                cov = np.zeros((d,d))
                for m in range(d):
                    cov[m,m] = covars[j,m]
                
                if hw5.p(pai[j], X[i,:], mu[j,:], cov, d) == 0:
                    a = 1.0e-298
                    log += R[i,j]*math.log(a/R[i,j],math.e)
                else:
                    log += R[i,j]*math.log(hw5.p(pai[j], X[i,:], mu[j,:], cov, d)/R[i,j],math.e)
    if singular == False:
        break
    score += [log]
print(score)
#hw5_utils.line_plot(score)


'''
[-387.2571434512502, -308.2493719702462, -265.91936837008524, -241.20460820963729, -221.5814137404429, -205.34419545431882, -190.0351206566011, -187.47147211698987,-197.4737652295963]
X, tex, Y, tey=hw5_utils.load_iris_data(ratio=1)
mu, covars, pai, R = hw5.gmm(X, 4)
n = X.shape[0]
A = np.zeros((n,4))
for i in range(n):
    j = np.argmax(R[i,:])
    A[i,j]=1
A = A.astype(np.int32)
j = []
for i in range(4):
    j += [np.where(A[:,i]==1)[0]]
hw5_utils.scatter_plot_2d_project(X[j[0],:], X[j[1],:], X[j[2],:], X[j[3],:], covars)
'''