#!/bin/python
# Version 2.0

import sys
import math
import numpy as np
from sklearn.datasets import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

def load_iris_data(ratio=1):
    """
    Loads the Iris dataset and splits it into testing and training datasets

    arguments:
    ratio -- the ratio of test set to the entire dataset

    return:
    X_train -- training dataset
    X_test -- test dataset
    Y_train -- training labels
    Y_test -- test labels
    """
    X, Y = load_iris(True)
    return train_test_split(X, Y, test_size=1-ratio)

def logistic_regression(X, Y):
    """
    Fits a logistic regression model to the given data

    arguments:
    X -- dataset
    Y -- labels

    return:
    an sklearn.linear_model.LogisticRegression object that has been trained on the dataset X, Y
    """
    return LogisticRegression(solver='lbfgs', multi_class='multinomial').fit(X, Y)

def line_plot(*data, min_k=2, output_file='output.pdf'):
    """
    Plots the line plot of the given data

    arguments:
    *data -- a list of data arrays to be plotted
    min_k -- the first index on the x axis
    output_file -- the location of the file to print the plot to
    """
    fig = plt.figure(figsize=(30, 10))
    for d in data:
        plt.plot(range(min_k, min_k + len(d)), d)
    #plt.xticks( range(min_k,21,1))
    plt.savefig(output_file)

def scatter_plot_2d_project(*data, output_file='output.pdf', ncol=3):
    """
    Plots the scatter plot of the given data

    arguments:
    *data -- a list of data matricies to be plotted
    output_file -- the location of the file to print the plot to

    example:
    If X is a data matrix of shape (n,d) and A is an assignment matrix of shape (n,k) and C is the matrix of centers of shape (k,d), you would write:
        scatter_plot_2d_project(X[A[:,0],:], X[A[:,1],:], X[A[:,2],:], X[A[:,3],:], C)
    """
    fig = plt.figure(figsize=(30, 10))
    d = data[0].shape[1]
    for i in range(d):
        for j in range(i):
            ax = plt.subplot(math.ceil(d*(d-1)/2/ncol), ncol, i*(i-1)//2+j+1)
            for X in data:
                plt.scatter(X[:,j], X[:,i])
            plt.xlabel('dim{}'.format(j))
            plt.ylabel('dim{}'.format(i))
    plt.savefig(output_file)

def gaussian_plot_2d_project(mu, variances, *data, output_file='output.pdf', ncol=3):
    """
    Plots the scatter plot of the given data

    arguments:
    *data -- a list of data matricies to be plotted
    output_file -- the location of the file to print the plot to

    example:
    If X is a data matrix of shape (n,d) and A is an assignment matrix of shape (n,k) and mu is the matrix of centers of shape (k,d) and variances is the variance matrix of shape(k,d), you would write:
        gaussian_plot_2d_project(mu, variances, X[A[:,0],:], X[A[:,1],:], X[A[:,2],:], X[A[:,3],:])
    """
    fig = plt.figure(figsize=(30, 10))
    d = data[0].shape[1]
    for i in range(d):
        for j in range(i):
            ax = plt.subplot(math.ceil(d*(d-1)/2/ncol), ncol, i*(i-1)//2+j+1)
            for (l, X) in enumerate(data):
                plt.scatter(X[:,j], X[:,i])
                if l < mu.shape[0] and l < variances.shape[0]:
                    (xmin, xmax, ymin, ymax) = (np.min(X[:,j]), np.max(X[:,j]), np.min(X[:,i]), np.max(X[:,i]))
                    xrange = np.arange(xmin, xmax, (xmax-xmin)/256)
                    yrange = np.arange(ymin, ymax, (ymax-ymin)/256)
                    xmesh, ymesh = np.meshgrid(xrange, yrange)
                    stack = np.dstack((xmesh, ymesh))
                    plt.contour(xmesh, ymesh, multivariate_normal.pdf(stack, mean=mu[l,[j,i]], cov=np.diag(variances[l,[j,i]])), cmap='Greys')
            plt.xlabel('dim{}'.format(j))
            plt.ylabel('dim{}'.format(i))
    plt.savefig(output_file)
