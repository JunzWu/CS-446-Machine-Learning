import numpy as np
import scipy
import scipy.spatial
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

def load_reg_data():
    # load the regression synthetic data
    np.random.seed(1) # force seed so same data is generated every time
    X = np.linspace(0,4,num=100)
    noise = np.random.normal(size=X.shape, scale=0.4)
    w = 0.5
    b = 1.
    Y = w * X**2 + b + noise
    X = np.reshape(X,[-1,1])
    return X,Y

def load_xor_data():
    X = np.array([[-1,1],[1,-1],[-1,-1],[1,1]])
    Y = np.prod(X,axis=1)
    return X,Y

def load_nn_data():
    np.random.seed(3)
    X1,Y1 = linear_problem(np.array([-1,0.3]), margin=1.0, size=100)
    X2,Y2 = linear_problem(np.array([1,-0.1]), margin=6.0, size=100, bounds=[-7,7],trans=3)
    X = np.concatenate([X1,X2], axis=0)
    Y = np.concatenate([Y1,Y2],axis=0)
    return X,Y

def load_iris_data():
    X,Y = load_iris(return_X_y=True)
    np.random.seed(1)
    s = np.arange(len(X))
    np.random.shuffle(s)
    return X[s],Y[s]

def load_logistic_data():
    np.random.seed(1) # reset seed
    return linear_problem(np.array([-1.,2.]),margin=1.5,size=200)


def voronoi_plot(X,Y):
    # takes as input data set and saves a voronoi plot
    voronoi = scipy.spatial.Voronoi(X)
    plt.clf()
    #render it once transparent so that the two figures have the points in the same places.
    #(to avoid "popping" when showing the two figures.)
    scipy.spatial.voronoi_plot_2d(voronoi, show_points = False, show_vertices = False,
                                  line_alpha = 0.0, line_width = 0.5,)
    plt.scatter(X[:, 0], X[:, 1], c = [ 'red' if yy >= 0 else 'blue' for yy in Y ], marker = 'X')
    plt.tight_layout()
    plt.savefig('1nn_data.pdf')
    scipy.spatial.voronoi_plot_2d(voronoi, show_points = False, show_vertices = False,
                                  line_alpha = 1.0, line_width = 0.5,)
    plt.scatter(X[:, 0], X[:, 1], c = [ 'red' if yy >= 0 else 'blue' for yy in Y ], marker = 'X',
                zorder = 4)
    plt.tight_layout()
    plt.savefig('1nn_voronoi.pdf')

def linear_problem(w,margin,size,bounds=[-5.,5.],trans=0.0):
    in_margin = lambda x: np.abs(w.dot(x)) / np.linalg.norm(w) < margin
    X = []
    Y = []
    for i in range(size):
        x = np.random.uniform(bounds[0],bounds[1],size=[2]) + trans
        while in_margin(x):
            x = np.random.uniform(bounds[0],bounds[1],size=[2]) + trans
        if w.dot(x) + trans > 0:
            Y.append(1.)
        else:
            Y.append(-1.)
        X.append(x)
    X = np.array(X)
    Y = np.array(Y)
    return X,Y
