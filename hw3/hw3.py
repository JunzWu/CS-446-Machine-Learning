import hw3_utils
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def svm_solver(x_train, y_train, lr, num_iters,
               kernel=hw3_utils.poly(degree=1), c=None):
    """An SVM solver.

    Arguments:
        x_train: a 2d tensor with shape (n, d).
        y_train: a 1d tensor with shape (n,), whose elememnts are +1 or -1.
        lr: the learning rate.
        num_iters: the number of gradient descent steps.
        kernel: the kernel function.
           The default kernel function is 1 + <x, y>.
        c: the trade-off parameter in soft-margin SVM.
           The default value is None, referring to the basic, hard-margin SVM.

    Return:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
               Initialize alpha to be 0.
               Return alpha.detach() could possibly help you save some time
               when you try to use alpha in other places.

    Note that if you use something like alpha = alpha.clamp(...) with
    torch.no_grad(), you will have alpha.requires_grad=False after this step. 
    You will then need to use alpha.requires_grad_().
    Alternatively, use in-place operations such as clamp_().
    """
    alpha = torch.zeros(x_train.size(0), requires_grad = True)
    for i in range(num_iters):
        f = 0
        for j in range(x_train.size(0)):
            f -= alpha[j]
            for k in range(x_train.size(0)):
                f += 0.5*alpha[j]*alpha[k]*y_train[j]*y_train[k]*kernel(x_train[j,:], x_train[k,:])

        f.backward()
        with torch.no_grad():
            alpha -= lr * alpha.grad
            if c == None:
                for j in range(x_train.size(0)):
                    alpha[j] = max(alpha[j], 0)
            else:
                for j in range(x_train.size(0)):
                    alpha[j] = min(max(alpha[j], 0), c)
            alpha.grad.zero_()

    return alpha.detach()


def svm_predictor(alpha, x_train, y_train, x_test,
                  kernel=hw3_utils.poly(degree=1)):
    """An SVM predictor.

    Arguments:
        alpha: a 1d tensor with shape (n,), denoting an optimal dual solution.
        x_train: a 2d tensor with shape (n, d), denoting the training set.
        y_train: a 1d tensor with shape (n,), whose elememnts are +1 or -1.
        x_test: a 2d tensor with shape (m, d), denoting the test set.
        kernel: the kernel function.
           The default kernel function is 1 + <x, y>.

    Return:
        A 1d tensor with shape (m,), the outputs of SVM on the test set.
    """
    y_test = torch.zeros(x_test.size(0))
    for i in range(x_test.size(0)):
        for j in range(x_train.size(0)):
            y_test[i] += alpha[j]*y_train[j]*kernel(x_train[j,:], x_test[i,:])
    return y_test


def svm_contour(alpha, x_train, y_train, kernel,
                xmin=-5, xmax=5, ymin=-5, ymax=5, ngrid = 33):
    """Plot the contour lines of the svm predictor. """
    with torch.no_grad():
        xgrid = torch.linspace(xmin, xmax, ngrid)
        ygrid = torch.linspace(ymin, ymax, ngrid)
        (xx, yy) = torch.meshgrid((xgrid, ygrid))
        x_test = torch.cat(
            (xx.view(ngrid, ngrid, 1), yy.view(ngrid, ngrid, 1)),
            dim = 2).view(-1, 2)
        zz = svm_predictor(alpha, x_train, y_train, x_test, kernel)
        zz = zz.view(ngrid, ngrid)
        cs = plt.contour(xx.cpu().numpy(), yy.cpu().numpy(), zz.cpu().numpy(),
                        cmap = 'RdYlBu')
        plt.clabel(cs)
        plt.show()


class Block(nn.Module):
    """A basic block used to build ResNet."""

    def __init__(self, num_channels):
        """Initialize a building block for ResNet.

        Argument:
            num_channels: the number of channels of the input to Block, and
                          the number of channels of conv layers of Block.
        """
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias = False)
        self.conv1_BN = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, 3, 1, 1, bias = False)
        self.conv2_BN = nn.BatchNorm2d(num_channels)
        
    def forward(self, x):
        """
        The input will have shape (N, num_channels, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have the same shape as input.
        """
        x1 = self.conv1(x)
        x1 = self.conv1_BN(x1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.conv2_BN(x1)
        x = F.relu(x+x1)
        return x

    def set_param(self, kernel_1, bn1_weight, bn1_bias,
                  kernel_2, bn2_weight, bn2_bias):
        """Set the parameters of self using given arguments.

        Parameters of a Conv2d, BatchNorm2d, and Linear 
        are all given by attributes weight and bias.
        Note that you should wrap the arguments in nn.Parameter.

        Arguments (C denotes number of channels):
            kernel_1: a (C, C, 3, 3) tensor, kernels of the first conv layer.
            bn1_weight: a (C,) tensor.
            bn1_bias: a (C,) tensor.
            kernel_2: a (C, C, 3, 3) tensor, kernels of the second conv layer.
            bn2_weight: a (C,) tensor.
            bn2_bias: a (C,) tensor.
        """
        self.conv1.weight = nn.Parameter(kernel_1)
        self.conv1_BN.weight = nn.Parameter(bn1_weight)
        self.conv1_BN.bias = nn.Parameter(bn1_bias)
        self.conv2.weight = nn.Parameter(kernel_2)
        self.conv2_BN.weight = nn.Parameter(bn2_weight)
        self.conv2_BN.bias = nn.Parameter(bn2_bias)
        
        pass


class ResNet(nn.Module):
    """A simplified ResNet."""

    def __init__(self, num_channels, num_classes=10):
        """Initialize a shallow ResNet.

        Arguments:
            num_channels: the number of output channels of the conv layer
                          before the building block, and also 
                          the number of channels of the building block.
            num_classes: the number of output units.
        """
        super(ResNet, self).__init__()
        self.conv0 = nn.Conv2d(1, num_channels, 3, 2, 1, bias = False)
        self.conv0_BN = nn.BatchNorm2d(num_channels)
        self.MP = nn.MaxPool2d(2)
        self.Block = Block(num_channels)
        self.AAP = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_channels, 10)

    def forward(self, x):
        """
        The input will have shape (N, 1, H, W),
        where N is the batch size, and H and W give the shape of each channel.

        The output should have shape (N, 10).
        """
        x = self.conv0(x)
        x = self.conv0_BN(x)
        x = F.relu(x)
        x = self.MP(x)
        x = self.Block(x)
        x = self.AAP(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        
        
        return x

    def set_param(self, kernel_0, bn0_weight, bn0_bias,
                  kernel_1, bn1_weight, bn1_bias,
                  kernel_2, bn2_weight, bn2_bias,
                  fc_weight, fc_bias):
        """Set the parameters of self using given arguments.

        Parameters of a Conv2d, BatchNorm2d, and Linear 
        are all given by attributes weight and bias.
        Note that you should wrap the arguments in nn.Parameter.

        Arguments (C denotes number of channels):
            kernel_0: a (C, 1, 3, 3) tensor, kernels of the conv layer
                      before the building block.
            bn0_weight: a (C,) tensor, weight of the batch norm layer
                        before the building block.
            bn0_bias: a (C,) tensor, bias of the batch norm layer
                      before the building block.
            fc_weight: a (10, C) tensor
            fc_bias: a (10,) tensor
        See the docstring of Block.set_param() for the description
        of other arguments.
        """
        self.conv0.weight = nn.Parameter(kernel_0)
        self.conv0_BN.weight = nn.Parameter(bn0_weight)
        self.conv0_BN.bias = nn.Parameter(bn0_bias)
        self.Block.set_param(kernel_1, bn1_weight, bn1_bias, kernel_2, bn2_weight, bn2_bias)
        self.fc.weight = nn.Parameter(fc_weight)
        self.fc.bias = nn.Parameter(fc_bias)
        
        pass