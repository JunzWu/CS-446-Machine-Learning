import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data
from sklearn.datasets import load_digits


def contour_torch(xmin, xmax, ymin, ymax, M, ngrid = 33):
    """
    make a contour plot without the magic

    Note- your network can be passed in as paramter M without any modification.
    @param xmin: lowest value of x in the plot
    @param xmax: highest value of x in the plot
    @param ymin: ditto for y
    @param ymax: ditto for y
    @param M: prediction function, takes a (X,Y,2) torch tensor as input and returns an (X,Y) torch tensor as output
    @param ngrid: 
    """
    with torch.no_grad():
        (X,Y) = XOR_data()
        xgrid = torch.linspace(xmin, xmax, ngrid)
        ygrid = torch.linspace(ymin, ymax, ngrid)
        (xx, yy) = torch.meshgrid((xgrid, ygrid))
        D = torch.cat((xx.reshape(ngrid, ngrid, 1), yy.reshape(ngrid, ngrid, 1)), dim = 2)
        zz = M(D)[:,:,0]
        cs = plt.contour(xx.cpu().numpy(), yy.cpu().numpy(), zz.cpu().numpy(),
                        cmap = 'RdYlBu')
        plt.clabel(cs)
        for i in range(Y.shape[0]):
            if Y[i] == 1:
                plt.plot(X[i,0],X[i,1],'ro')
            if Y[i] == 0:
                plt.plot(X[i,0],X[i,1],'bo')
        plt.show()


def torch_digits():
    """
    Get the training and test datasets for your convolutional neural network
    @return train, val: two torch.utils.data.Datasets
    """
    digits, labels = load_digits(return_X_y=True)
    digits = torch.tensor(np.reshape(digits, [-1, 8, 8]), dtype=torch.float)
    print(digits.shape)
    labels = torch.tensor(np.reshape(labels, [-1]), dtype=torch.long)
    val_X = digits[:180,:,:]
    val_Y = labels[:180]
    digits = digits[180:,:,:]
    labels = labels[180:]
    train = torch.utils.data.TensorDataset(digits, labels)
    val = torch.utils.data.TensorDataset(val_X, val_Y)
    return train, val


def XOR_data():
    X = torch.tensor([[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]])
    Y = (-torch.prod(X, dim=1)+1.)/2 
    return X, Y.view(-1,1)


def plot_PCA(intermediate, labels):
    """
    Create a scatterplot of intermediate 
    @param intermediate: numpy NxD
    @param labels: numpy (N,)
    """
    pca = PCA(2)
    ft = pca.fit_transform(intermediate)
    for i in range(10):
        plt.scatter(ft[labels==i,0], ft[labels==i, 1], label=str(i), alpha=0.4)
    plt.legend()
    plt.show()


def get_image():
    """
    @return img: (N, M, 3) image with values ranging from 0 to 1
    """
    return plt.imread('LunarEclipseCologne_Junius_960.jpg')/255.0 # display image as a float to avoid overflows/underflows


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
