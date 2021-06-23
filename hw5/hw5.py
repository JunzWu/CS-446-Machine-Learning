 #!/bin/python
#Version 2.0

import numpy as np
import torch
import hw5_utils
from scipy.stats import norm
import math

################################# Problem 2 #################################
def k_means(X, k):
    """
    Implements Lloyd's algorithm.

    arguments:
    X -- n by d data matrix
    k -- integer, number of centers

    return:
    A matrix C of shape k by d with centers as its rows.
    """
    #Hint: You can use np.random.randn to initialize the centers randomly.
    #Hint: Implement auxiliary functions for recentering and for reassigning. Then repeat until no change.
    n = X.shape[0]
    d = X.shape[1]
    maximum = []
    minimum = []
    for i in range(d):
        maximum += [np.max(X[:,i])]
        minimum += [np.min(X[:,i])]
    C = np.zeros((k,d))
    for i in range(d):
        C[:,i] = np.random.uniform(minimum[i], maximum[i], k)
    A = np.zeros((n,k))
    for i in range(n):
        distance = []
        for j in range(k):
            distance += [np.linalg.norm(X[i,:]-C[j,:])]
        j = distance.index(min(distance))
        A[i,j]=1
    #print(A)
    phi = 0
    for i in range(n):
        for j in range(k):
            phi += A[i,j]*np.dot(X[i,:]-C[j,:],X[i,:]-C[j,:])
    #print(phi)
    num = 0
    while True:
        for i in range(k):
            points = X[np.where(A[:,i] == 1)[0]]
            if points.shape[0] != 0:
                C[i,:] = np.mean(points,axis=0)
            else:
                C[i,:] = np.random.randn(d)
        
        A = np.zeros((n,k))
        for i in range(n):
            distance = []
            for j in range(k):
                distance += [np.linalg.norm(X[i,:]-C[j,:])]
            j = distance.index(min(distance))
            A[i,j]=1
        #print(A)
        phi1 = 0
        for i in range(n):
            for j in range(k):
                phi1 += A[i,j]*np.dot(X[i,:]-C[j,:],X[i,:]-C[j,:])
        
        delta = phi - phi1
        #print(phi1)
        if delta < 0.00000005 and num == 10:
            break
        if delta < 0.00000005 and num < 10:
            phi = phi1
            num += 1
        if delta >= 0.00000005:
            phi = phi1
    
    return C

def get_purity_score(X, Y, C):
    """
    Computes the purity score for each cluster.

    arguments:
    X -- n by d data matrix
    Y -- n by 1 label vector
    C -- k by d center matrix

    return:
    Fraction of points with label matching their cluster's majority label.
    """
    n = X.shape[0]
    k = C.shape[0]
    r = 0
    
    A = np.zeros((n,k))
    for i in range(n):
        distance = []
        for j in range(k):
            distance += [np.linalg.norm(X[i,:]-C[j,:])]
        j = distance.index(min(distance))
        A[i,j]=1
    
    for i in range(k):
        y = Y[np.where(A[:,i]==1)[0]]
        a = np.zeros(3)
        for j in range(y.shape[0]):
            if y[j] == 0:
                a[0] += 1
            if y[j] == 1:
                a[1] += 1
            if y[j] == 2:
                a[2] += 1
        r += np.max(a)
    return r/n

def classify_using_k_means(X, Y, k, l=1):
    """
    Classifies the datapoints learning features by k-means and classifying by logistic regression.

    arguments:
    X -- n by d data matrix
    Y -- n by 1 label vector
    k -- integer; number of components
    l -- integer; number of centers to take into account

    return:
    lr -- a logistic classifier
    C -- k by d matrix of centers
    
    assertions:
    l <= k
    """
    assert l <= k, 'k should larger than l'
    C = k_means(X, k)
    n = X.shape[0]
    A2 = np.zeros((n,k))
    
    for i in range(n):
        distance1 = np.zeros(k)
        distance2 = np.zeros(k)
        for j in range(k):
            distance1[j] = np.linalg.norm(X[i,:]-C[j,:])
            distance2[j] = np.linalg.norm(X[i,:]-C[j,:])
        distance2 = distance2[np.argsort(distance2)]
        
        for a in range(l):
            j = np.where(distance1 == distance2[a])[0]
            for b in range(j.shape[0]):
                if A2[i,j[b]] == 0:
                    A2[i,j[b]] = 1
                    break
           
    lr = hw5_utils.logistic_regression(A2, Y)
    
    return lr, C
    


################################# Problem 3 #################################
def p(pai, x, mu, cov, d):
    p1 = pai / math.sqrt(np.power(2*np.pi,d)*np.linalg.det(cov))
    diff = (x - mu).reshape((-1,1))
    p2 = math.exp(-0.5*np.dot(np.dot(diff.T, np.linalg.inv(cov)),diff))
    return p1*p2

def gmm(X, k, epsilon=0.0001):
    """
    Computes the maximum likelihood Gaussian mixture model using expectation maximization algorithm.

    argument:
    X -- n by d data matrix
    k -- integer; number of Gaussian components
    epsilon -- improvement lower bound

    return:
    mu -- k by d matrix with centers as rows
    covars -- a list of k covariance matrices of shape (d, d)
    weights -- k by 1 vector of probabilities over the Gaussian components
    """
    n = X.shape[0]
    d = X.shape[1]
    mu = k_means(X, k)
    cov = []
    for i in range(k):
        cov += [np.eye(d)]
    pai = np.ones(k)/k
    mu_old = mu.copy()
    #print(mu)
    while True:
        #E-step
        R = np.zeros((n,k))
        for i in range(n):
            for j in range(k):
                s = 0
                for l in range(k):
                    s += p(pai[l], X[i,:], mu[l,:], cov[l], d)
                R[i,j] = p(pai[j], X[i,:], mu[j,:], cov[j], d)/s
        
        #M-step
        for j in range(k):
            
            #update pai
        
            s = 0
            for i in range(n):
                s += R[i,j]
            pai[j] = s/n
            
            #update mu
            s = np.zeros(d)
            for i in range(n):
                s += R[i,j]*X[i,:]
            mu[j,:] = s/(n*pai[j])
            #update covariance matrix
            s = np.zeros((d,d))
            for i in range(n):
                diff = (X[i,:] - mu[j,:]).reshape((-1,1))
                s += R[i,j]*np.dot(diff, diff.T)
            cov[j] = s/(n*pai[j])
            
            cov1 = cov[j].diagonal()
            cov[j] = np.zeros((d,d))
            for l in range(d):
                cov[j][l,l] = cov1[l]

            
        #calculate the change
        diff = mu - mu_old
        change = []
        for i in range(k):
            change += [np.linalg.norm(diff[i,:])]
        change_max = max(change)
        #print(change_max)
        if change_max <= epsilon:
            
            break
        else:
            mu_old = mu.copy()
    
    covars = np.zeros((k,d))
    for i in range(k):
        covars[i,:] = cov[i].diagonal() 
        
    return mu, covars, pai

def gmm_predict(x, mu, covars, weights):
    """
    Computes the posterior probability of x having been generated by each of the k Gaussian components.

    arguments:
    x -- a single data point
    mu -- k by d matrix of centers
    covars -- a list k covariance matrices of shape (d, d)
    weights -- k by 1 vector of probabilities over the Gaussian components

    return:
    a k-vector that is the probability distribution of x having been generated by each of the Gaussian components.
    """
    k = mu.shape[0]
    d = mu.shape[1]
    P = np.zeros(k)
    for i in range(k):
        s = 0
        for l in range(k):
            cov = np.zeros((d,d))
            for m in range(d):
                cov[m,m] = covars[l,m]
            s += p(weights[l], x, mu[l,:], cov, d)
            
        cov = np.zeros((d,d))
        for j in range(d):
            cov[j,j] = covars[i,j]
        P[i] = p(weights[i], x, mu[i,:], cov, d)/s
    
    return P
def classify_using_gmm(X, Y, k):
    """
    Classifies the datapoints learning features by GMM and classifying by logistic regression.

    arguments:
    X -- n by d data matrix
    Y -- n by 1 label vector
    k -- integer; number of components

    return:
    lr -- a logistic classifier
    mu -- k by d matrix of centers
    variances -- k by d matrix of variances
    weights -- k-vector of component weights
    """
    mu, covars, pai = gmm(X, k)
    n = X.shape[0]
    d = X.shape[1]
    R = np.zeros((n,k))
    for i in range(n):
        for j in range(k):
            s = 0
            for l in range(k):
                cov = np.zeros((d,d))
                for m in range(d):
                    cov[m,m] = covars[l,m]
                s += p(pai[l], X[i,:], mu[l,:], cov, d)
            cov = np.zeros((d,d))
            for m in range(d):
                cov[m,m] = covars[j,m]
            R[i,j] = p(pai[j], X[i,:], mu[j,:], cov, d)/s
           
    lr = hw5_utils.logistic_regression(R, Y)
    
    return lr, mu, covars, pai
