# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 12:39:14 2019

@author: Junz
"""

import matplotlib.pyplot as plt
import numpy
import torch

def contour_numpy(xmin, xmax, ymin, ymax, M, ngrid = 33):
    """
    make a contour plot without 
    @param xmin: lowest value of x in the plot
    @param xmax: highest value of x in the plot
    @param ymin: ditto for y
    @param ymax: ditto for y
    @param M: prediction function, takes a (X,Y,2) numpy ndarray as input and returns an (X,Y) numpy ndarray as output
    @param ngrid: 
    """
    (X,Y) = utils.load_logistic_data()
    xgrid = numpy.linspace(xmin, xmax, ngrid)
    ygrid = numpy.linspace(ymin, ymax, ngrid)
    (xx, yy) = numpy.meshgrid(xgrid, ygrid)
    D = numpy.dstack((xx, yy)) # D is (X, Y, 2)
    zz = M(D)
    C = plt.contour(xx, yy, zz, cmap = 'rainbow')
    plt.clabel(C)
    for i in range(Y.shape[0]):
        if Y[i] == 1:
            plt.plot(X[i,0],X[i,1],'ro')
        if Y[i] == -1:
            plt.plot(X[i,0],X[i,1],'bo')
    plt.show()
    
def contour_torch(xmin, xmax, ymin, ymax, M, ngrid = 33):
    """
    make a contour plot without 
    @param xmin: lowest value of x in the plot
    @param xmax: highest value of x in the plot
    @param ymin: ditto for y
    @param ymax: ditto for y
    @param M: prediction function, takes a (X,Y,2) torch tensor as input and returns an (X,Y) torch tensor as output
    @param ngrid: 
    """
    xgrid = torch.linspace(xmin, xmax, ngrid)
    ygrid = torch.linspace(ymin, ymax, ngrid)
    (xx, yy) = torch.meshgrid((xgrid, ygrid))
    D = torch.cat((xx.reshape(ngrid, ngrid, 1), yy.reshape(ngrid, ngrid, 1)), dim = 2)
    zz = M(D)
    C = plt.contour(xx.cpu().numpy(), yy.cpu().numpy(), zz.cpu().numpy(), cmap = 'rainbow')
    plt.clabel(C)
    plt.show()

import hw1_utils as utils
import hw1

def M1(x):
    (X,Y) = utils.load_logistic_data()
    w = hw1.linear_normal(X,Y)
    return w[0] + w[1]*x[:,:,0] + w[2]*x[:,:,1]

def M2(x):
    (X,Y) = utils.load_logistic_data()
    w = hw1.logistic(X,Y)
    return w[0]*x[:,:,0] + w[1]*x[:,:,1]

contour_numpy(-6, 6, -6, 6, M1)
contour_numpy(-6, 6, -6, 6, M2)