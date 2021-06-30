import hw6_utils
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable


class VAE(torch.nn.Module):
    def __init__(self, lam,lrate,latent_dim,loss_fn):
        """
        Initialize the layers of your neural network

        @param lam: Hyperparameter to scale KL-divergence penalty
        @param lrate: The learning rate for the model.
        @param loss_fn: A loss function defined in the following way:
            @param x - an (N,D) tensor
            @param y - an (N,D) tensor
            @return l(x,y) an () tensor that is the mean loss
        @param latent_dim: The dimension of the latent space

        The network should have the following architecture (in terms of hidden units):
        Encoder Network:
        2 -> 50 -> ReLU -> 50 -> ReLU -> 50 -> ReLU -> (6,6) (mu_layer,logstd2_layer)

        Decoder Network:
        6 -> 50 -> ReLU -> 50 -> ReLU -> 2 -> Sigmoid

        See set_parameters() function for the exact shapes for each weight
        """
        super(VAE, self).__init__()

        self.lrate = lrate
        self.lam = lam
        self.loss_fn = loss_fn
        self.latent_dim = latent_dim
        
        self.en_fc1 = nn.Linear(2,50)
        self.en_fc2 = nn.Linear(50,50)
        self.en_fc3 = nn.Linear(50,50)
        self.en_mu = nn.Linear(50,latent_dim)
        self.en_log = nn.Linear(50,latent_dim)
        
        self.de_fc1 = nn.Linear(latent_dim,50)
        self.de_fc2 = nn.Linear(50,50)
        self.de_fc3 = nn.Linear(50,2)
        
        self.opt = optim.Adam(self.parameters(), lr=self.lrate)

    def set_parameters(self, We1,be1, We2, be2, We3, be3, Wmu, bmu, Wstd, bstd, Wd1, bd1, Wd2, bd2, Wd3, bd3):
        """ Set the parameters of your network

        # Encoder weights:
        @param We1: an (50,2) torch tensor
        @param be1: an (50,) torch tensor
        @param We2: an (50,50) torch tensor
        @param be2: an (50,) torch tensor
        @param We3: an (50,50) torch tensor
        @param be3: an (50,) torch tensor
        @param Wmu: an (6,50) torch tensor
        @param bmu: an (6,) torch tensor
        @param Wstd: an (6,50) torch tensor
        @param bstd: an (6,) torch tensor

        # Decoder weights:
        @param Wd1: an (50,6) torch tensor
        @param bd1: an (50,) torch tensor
        @param Wd2: an (50,50) torch tensor
        @param bd2: an (50,) torch tensor
        @param Wd3: an (2,50) torch tensor
        @param bd3: an (2,) torch tensor

        """
        self.en_fc1.weight=nn.Parameter(We1)
        self.en_fc1.bias=nn.Parameter(be1)
        
        self.en_fc2.weight=nn.Parameter(We2)
        self.en_fc2.bias=nn.Parameter(be2)
        
        self.en_fc3.weight=nn.Parameter(We3)
        self.en_fc3.bias=nn.Parameter(be3)
        
        self.en_mu.weight=nn.Parameter(Wmu)
        self.en_mu.bias=nn.Parameter(bmu)
        
        self.en_log.weight=nn.Parameter(Wstd)
        self.en_log.bias=nn.Parameter(bstd)
        
        self.de_fc1.weight=nn.Parameter(Wd1)
        self.de_fc1.bias=nn.Parameter(bd1)
        
        self.de_fc2.weight=nn.Parameter(Wd2)
        self.de_fc2.bias=nn.Parameter(bd2)
        
        self.de_fc3.weight=nn.Parameter(Wd3)
        self.de_fc3.bias=nn.Parameter(bd3)
        
        return 

    def forward(self, x):
        """ A forward pass of your autoencoder

        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return y: an (N, 50) torch tensor of output from the encoder network
        @return mean: an (N,latent_dim) torch tensor of output mu layer
        @return stddev_p: an (N,latent_dim) torch tensor of output stddev layer
        @return z: an (N,latent_dim) torch tensor sampled from N(mean,exp(stddev_p/2)
        @return xhat: an (N,D) torch tensor of outputs from f_dec(z)
        """
        y = self.en_fc1(x)
        y = F.relu(y)
        y = self.en_fc2(y)
        y = F.relu(y)
        y = self.en_fc3(y)
        y = F.relu(y)

        mean = self.en_mu(y)
        stddev_p = self.en_log(y)
        
        n = x.shape[0]
        z = torch.randn(n,self.latent_dim)
        std = torch.exp(stddev_p/2.0)
        z = z.mul(std) + mean
        
        xhat = self.de_fc1(z)
        xhat = F.relu(xhat)
        xhat = self.de_fc2(xhat)
        xhat = F.relu(xhat)
        xhat = self.de_fc3(xhat)
        xhat = F.sigmoid(xhat)
        
        return y,mean,stddev_p,z,xhat
    
    def decoding(self, z):
        
        xhat = self.de_fc1(z)
        xhat = F.relu(xhat)
        xhat = self.de_fc2(xhat)
        xhat = F.relu(xhat)
        xhat = self.de_fc3(xhat)
        xhat = F.sigmoid(xhat)
        
        return xhat
    
    def step(self, x):
        """
        Performs one gradient step through a batch of data x
        @param x: an (N, 2) torch tensor

        # return the following in this order from left to right:
        @return L_rec: float containing the reconstruction loss at this time step
        @return L_kl: kl divergence penalty at this time step
        @return L: total loss at this time step
        """
        y,mean,stddev_p,z,xhat=self.forward(x)
        L_rec=self.loss_fn(x,xhat)

        var = torch.exp(stddev_p)
        s = torch.log(var) - torch.pow(mean,2) - var
        s = self.latent_dim + torch.sum(s, 1)
        s = -0.5*s
        s = self.lam*s       
        L_kl = s.mean()
        
        
        L=L_kl+L_rec

        L.backward()
        self.opt.step()
        self.opt.zero_grad()
        
        return L_rec,L_kl,L


def fit(net,X,n_iter):
    """ Fit a VAE.  Use the full batch size.
    @param net: the VAE
    @param X: an (N, D) torch tensor
    @param n_iter: int, the number of iterations of training

    # return all of these from left to right:

    @return losses_rec: Array of reconstruction losses at the beginning and after each iteration. Ensure len(losses_rec) == n_iter
    @return losses_kl: Array of KL loss penalties at the beginning and after each iteration. Ensure len(losses_kl) == n_iter
    @return losses: Array of total loss at the beginning and after each iteration. Ensure len(losses) == n_iter
    @return Xhat: an (N,D) NumPy array of approximations to X
    @return gen_samples: an (N,D) NumPy array of N samples generated by the VAE
    """

    losses_rec = np.zeros(n_iter)
    losses_kl = np.zeros(n_iter)
    losses = np.zeros(n_iter)
    
    for i in range(n_iter):
        L_rec,L_kl,L = net.step(X)
        losses_rec[i] = L_rec.detach().numpy()
        losses_kl[i] = L_kl.detach().numpy()
        losses[i] = L.detach().numpy()
    
    net.eval()
    y,mean,stddev_p,z,xhat=net.forward(X)
    
    n = X.shape[0]
    z = torch.randn(n, net.latent_dim)
    gen_samples=net.decoding(z)

    return losses_rec,losses_kl, losses, xhat.detach().numpy(), gen_samples.detach().numpy()
