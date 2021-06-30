# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 14:44:01 2019

@author: hasee
"""

import hw6
import hw6_utils
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np 


def l1_loss(x,y):
    return torch.abs(x-y).sum(1).mean()


loss_function=l1_loss
X=hw6_utils.generate_data()
net=hw6.VAE(0.001, 0.01, 6, loss_function)
losses_rec,losses_kl, losses, Xhat, gen_samples=hw6.fit(net,X,8000)



X=X.data.numpy()

plt.scatter(X[:,0],X[:,1],marker='o',c='b',label='x')
plt.scatter(gen_samples[:,0],gen_samples[:,1],marker='+',c='g',label='gen_samples')
plt.scatter(Xhat[:,0],Xhat[:,1],marker='x',c='r',label='xhat')
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

'''
x=np.arange(8000)
plt.plot(x,losses,'b',label='loss')
plt.legend()
plt.xlabel("epoch")
plt.ylabel('loss')
plt.show()
'''
#torch.save(net.cpu().state_dict(),"vae.pb")
