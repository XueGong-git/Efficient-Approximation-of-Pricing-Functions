### Generate uniform grid of training data given D, func, Nd. 
import numpy as np

def train(D, func, Nd, bounds):
    X = np.linspace(bounds[0] * np.ones(D), bounds[1] * np.ones(D), Nd) # Each column is list of grid-points for each dimension.
    Xs = np.hsplit(X, D) # Just for syntax: change array to tuple of vectors.
    XDarray = np.array(np.meshgrid(*Xs)) # Has shape (D, N,..., N), w. D axes of length Nd.
    
    Yarray = np.zeros(np.repeat(Nd, D))
    Yarray = func(XDarray, D)
    
    X_train = np.reshape(XDarray, (D, Nd ** D)) # Store coordinates of training points in a "list".
    Y_train = np.reshape(Yarray, Nd ** D) # Store training point function evaluations in a "list".
    N_train = Nd ** D # Total number of training points.
    
    Y_train = func(X_train, D)# * np.ones((1, Y_train.shape[0])) # np.ones is just to give Y_train second axis here.
    
    return(X_train, Y_train, N_train)