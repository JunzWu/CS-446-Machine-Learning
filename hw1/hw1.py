import numpy as np
import hw1_utils as utils
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Problem 2
def linear_gd(X,Y,lrate=0.1,num_iter=1000):
    # return parameters as numpy array
    w = np.zeros(X.shape[1]+1)
    x = np.ones((X.shape[0],1))
    X = np.hstack([x,X])
    for i in range(num_iter):
        g = np.dot(X.transpose(),np.dot(X,w)) - np.dot(X.transpose(),Y)
        g = g / X.shape[0]
        w -= lrate * g 
    return w




def linear_normal(X,Y):
    # return parameters as numpy array
    x = np.ones((X.shape[0],1))
    X = np.hstack([x,X])
    (U, S, VT) = np.linalg.svd(X)
    V = VT.transpose()
    Xz = np.zeros((X.shape[1], X.shape[0]))
    for i in range(S.shape[0]):
        Xz += 1/S[i]*np.dot(V[:,i].reshape(X.shape[1],1),U[:,i].reshape(1,X.shape[0]))
    w = np.dot(Xz, Y)
    return w

def plot_linear():
    # return plot
    (X,Y) = utils.load_reg_data()
    w = linear_gd(X,Y)
    print(w)
    x = np.ones((X.shape[0],1))
    X1 = np.hstack([x,X])
    y = np.dot(X1,w)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(X,Y,'ro')
    plt.plot(X,y)
    return plt.show()

# Problem 4
def poly_gd(X,Y,lrate=0.01,num_iter=3000):
    # return parameters as numpy array
    w = np.zeros((1+X.shape[1]+int((1+X.shape[1])*X.shape[1]/2),1))
    x1 = np.ones((X.shape[0],1))
    x2 = np.zeros((X.shape[0],int((1+X.shape[1])*X.shape[1]/2)))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]-i):
            x2[:,int((X.shape[1]+X.shape[1]-i+1)*i/2)+j] = X[:,i]*X[:,j+i]
    
    X = np.hstack([x1,X])
    X = np.hstack([X,x2])
    for i in range(num_iter):
        g = np.dot(X.transpose(),np.dot(X,w)) - np.dot(X.transpose(),Y).reshape(X.shape[1],1)
        g = g / X.shape[0]
        w -= lrate * g
    return w

def poly_normal(X,Y):
    # return parameters as numpy array
    x1 = np.ones((X.shape[0],1))
    x2 = np.zeros((X.shape[0],int((1+X.shape[1])*X.shape[1]/2)))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]-i):
            x2[:,int((X.shape[1]+X.shape[1]-i+1)*i/2)+j] = X[:,i]*X[:,j+i]
    
    X = np.hstack([x1,X])
    X = np.hstack([X,x2])

    (U, S, VT) = np.linalg.svd(X)
    V = VT.transpose()
    Xz = np.zeros((X.shape[1], X.shape[0]))
    for i in range(S.shape[0]):
        Xz += 1/S[i]*np.dot(V[:,i].reshape(X.shape[1],1),U[:,i].reshape(1,X.shape[0]))
    w = np.dot(Xz, Y)

    return w

def plot_poly():
    # return plot
    (X,Y) = utils.load_reg_data()
    w = poly_gd(X,Y)
    print(w)
    x1 = np.ones((X.shape[0],1))
    x2 = np.zeros((X.shape[0],int((1+X.shape[1])*X.shape[1]/2)))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]-i):
            x2[:,int((X.shape[1]+X.shape[1]-i+1)*i/2)+j] = X[:,i]*X[:,j+i]
    X1 = np.hstack([x1,X])
    X1 = np.hstack([X1,x2])
    y = np.dot(X1,w)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.plot(X,Y,'ro')
    plt.plot(X,y)
    plt.show()
    return


def poly_xor():
    # return labels for XOR from linear,polynomal models
    (X,Y) = utils.load_xor_data()
    w1 = linear_normal(X,Y)
    x = np.ones((X.shape[0],1))
    X1 = np.hstack([x,X])
    y_linear = np.dot(X1,w1)
    print(w1)
    w2 = poly_normal(X,Y)
    print(w2)
    x1 = np.ones((X.shape[0],1))
    x2 = np.zeros((X.shape[0],int((1+X.shape[1])*X.shape[1]/2)))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]-i):
            x2[:,int((X.shape[1]+X.shape[1]-i+1)*i/2)+j] = X[:,i]*X[:,j+i]
    X1 = np.hstack([x1,X])
    X1 = np.hstack([X1,x2])
    y_poly = np.dot(X1,w2)
    return y_linear,y_poly


# Problem 5
def nn(X,Y,X_test):
    # return labels for X_test as numpy array
    l = np.zeros(X_test.shape[0])
    for i in range(X_test.shape[0]):
        d = np.zeros(X.shape[0])
        for j in range(X.shape[0]):
            for k in range(X.shape[1]):
                d[j] += (X_test[i,k]-X[j,k])**2
        n = np.where(d==np.min(d))
        n = np.array(n)
        if n.shape[1] > 1:
            n = n[:,0]
        l[i] = Y[n]
    return l

def nn_iris():
    (X,Y) = utils.load_iris_data()
    X_test = X[:int(X.shape[0]*0.3),:]
    Y_true = Y[:int(X.shape[0]*0.3)]
    X1 = X[int(X.shape[0]*0.3):,:]
    Y1 = Y[int(X.shape[0]*0.3):]
    X1 = np.array(X1)
    Y1 = np.array(Y1)
    X_test = np.array(X_test)
    Y_result = nn(X1,Y1,X_test)
    j = 0
    for i in range(Y_result.shape[0]):
        if Y_result[i] != Y_true[i]:
            j += 1
    a = j/Y_result.shape[0]
    return 1-a


def loss(X, Y, w):
    torch.set_default_dtype(torch.float64)
    w = w.reshape(X.shape[1],1)
    l = torch.log(torch.exp(-X.mm(w) * Y)+1)
    return l

# Problem 6
def logistic(X,Y,lrate=1,num_iter=3000):
    # return parameters as numpy array
    torch.set_default_dtype(torch.float64)
    w = torch.zeros(X.shape[1], requires_grad = True)
    X = torch.from_numpy(X)
    Y = torch.from_numpy(Y).reshape(-1,1)
    for i in range(num_iter):
        l = loss(X, Y, w).mean()
        l.backward()
        with torch.no_grad():
            w -= lrate * w.grad
            w.grad.zero_()
    
    return w.detach().numpy()     

def logistic_vs_ols():
    # return plot
    X,Y = utils.load_logistic_data()
    w1 = logistic(X,Y)   
    w2 = linear_gd(X,Y)
    print(w1)
    print(w2)
    neg_data = np.array([X[i] for i in range(len(X)) if Y[i] == -1])
    pos_data = np.array([X[i] for i in range(len(X)) if Y[i] == 1])
    
    plt.scatter(neg_data[:,0], neg_data[:,1], color = 'red')
    plt.scatter(pos_data[:,0], pos_data[:,1], color = 'blue')
    
    X = np.linspace(-5, 5, num = 10)
    plt.plot(X, -(w1[0] / w1[1] * X))
    plt.plot(X, w2[0] / w2[2] -(w2[1] / w2[2] * X))

    return plt.show()

if __name__ == '__main__':
    logistic_vs_ols()