# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:40:48 2019

@author: hasee
"""

import hw2_utils as ut
import hw2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import spatial
from collections import Counter

'''
X,Y=ut.XOR_data()

net=hw2.XORNet()
sgd=torch.optim.SGD(net.parameters(),lr=0.005)
epoch_loss=hw2.fit(net,sgd,X,Y,5000)
ut.contour_torch(-5,5,-5,5,net)
'''
#p4
net=hw2.DigitsConvNet()

#train,val=ut.torch_digits()

x_train = np.load("data/x_train.npy")
x_train = (x_train - np.mean(x_train, axis=0)) / np.std(x_train, axis=0)
y_train = np.load("data/y_train.npy")
x_train = torch.tensor(np.reshape(x_train, [-1, 28, 28]), dtype=torch.float)
y_train = torch.tensor(np.reshape(y_train, [-1]), dtype=torch.long)

x_test = np.load("data/x_test.npy")
x_test = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
y_test = np.load("data/y_test.npy")
x_test = torch.tensor(np.reshape(x_test, [-1, 28, 28]), dtype=torch.float)
y_test = torch.tensor(np.reshape(y_test, [-1]), dtype=torch.long)

train = torch.utils.data.TensorDataset(x_train, y_train)
val = torch.utils.data.TensorDataset(x_test, y_test)

loss_fn = nn.CrossEntropyLoss()
sgd=torch.optim.SGD(net.parameters(),lr=0.005)
aa = torch.optim.lr_scheduler.ExponentialLR(sgd, 0.95)

train_el1 = []
val_el1 = []
for i in range(30):
    train_el,val_el=hw2.fit_and_validate(net,sgd,loss_fn,train,val,1)
    aa.step()
    if i == 0:
        train_el1 += [train_el[0]]
        train_el1 += [train_el[1]]
        val_el1 += [val_el[0]]
        val_el1 += [val_el[1]]
    if i != 0:
        train_el1 += [train_el[1]]
        val_el1 += [val_el[1]]
        
print(train_el1,val_el1)

x=np.arange(31)
plt.plot(x,train_el1,'r',label='train_epoch_loss')
plt.plot(x,val_el1,'b',label='val_epoch_loss')
plt.legend()
plt.xlabel("epoch")
plt.ylabel('loss')
plt.show()
'''
x_=np.zeros((1,4))
y_=np.zeros((1,1))
train_dl = torch.utils.data.DataLoader(train, 1)
for X,Y in train_dl:
    Y=Y.data.numpy()
    X=net.intermediate(X)
    X=X.data.numpy()
    Y=np.expand_dims(Y,axis=0)
    x_=np.vstack((x_,X))
    y_=np.vstack((y_,Y))

x_=np.delete(x_,0,0)
y_=np.delete(y_,0,0)

y_=y_.reshape(-1)

tree=spatial.KDTree(x_)

valx=[]
valy=[]
val_dl=torch.utils.data.DataLoader(val)
counts=0
error_count=0
for X,Y in val_dl:
    Y=Y.data.numpy()
    X=net.intermediate(X)
    X=X.data.numpy()
    distances,indexs=tree.query(X,5)
    predy=[]
    indexs=indexs[0]
    for i in indexs:
        predy.append(y_[i])
    predy_counts=Counter(predy)
    predy_=predy_counts.most_common(1)[0][0]
    if predy_ !=Y[0]:
        error_count+=1
    counts+=1
    
accuracy=1-float(error_count)/float(counts)
print(accuracy)

ut.plot_PCA(x_,y_)
'''

