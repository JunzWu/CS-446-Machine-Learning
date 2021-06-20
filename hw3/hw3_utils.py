import numpy as np
import torch
import torch.utils.data
from sklearn.datasets import load_digits


def poly_implementation(x, y, degree):
    assert x.size() == y.size(), 'The dimensions of inputs do not match!'
    with torch.no_grad():
        return (1 + (x * y).sum()).pow(degree)


def poly(degree):
    return lambda x, y: poly_implementation(x, y, degree)


def rbf_implementation(x, y, sigma):
    assert x.size() == y.size(), 'The dimensions of inputs do not match!'
    with torch.no_grad():
        return (-(x - y).norm().pow(2) / 2 / sigma / sigma).exp()


def rbf(sigma):
    return lambda x, y: rbf_implementation(x, y, sigma)


def xor_data():
    x = torch.tensor([[1, 1], [-1, 1], [-1, -1], [1, -1]], dtype=torch.float)
    y = torch.tensor([1, -1, 1, -1], dtype=torch.float)
    return x, y


def torch_digits():
    """
    Get the training and test datasets for your convolutional neural network
    @return train, val: two torch.utils.data.Datasets
    """
    digits, labels = load_digits(return_X_y=True)
    digits = torch.tensor(np.reshape(digits, [-1, 8, 8]), dtype=torch.float).unsqueeze(1)
    labels = torch.tensor(np.reshape(labels, [-1]), dtype=torch.long)
    val_X = digits[:180,:,:,:]
    val_Y = labels[:180]
    digits = digits[180:,:,:,:]
    labels = labels[180:]
    train = torch.utils.data.TensorDataset(digits, labels)
    val = torch.utils.data.TensorDataset(val_X, val_Y)
    return train, val


def loss_batch(model, loss_func, xb, yb, opt=None):
    """ Compute the loss of the model on a batch of data, or do a step of optimization.

    @param model: the neural network
    @param loss_func: the loss function (can be applied to model(xb), yb)
    @param xb: a batch of the training data to input to the model
    @param yb: a batch of the training labels to input to the model
    @param opt: a torch.optimizer.Optimizer.  If not None, use the Optimizer to improve the model. Otherwise, just compute the loss.
    @return a numpy array of the loss of the minibatch, and the length of the minibatch
    """
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb) 