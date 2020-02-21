### Implement GP regression, given D, training data, Nd, and bounds, to obtain 
### interpolated values for test points sampled uniformly at random, then 
### return the error for convergence tests. Uses specific cov. kernel.

import numpy as np
import numpy.linalg as linalg

def reg(D, func, X_train, Y_train, N_test, bounds):
    X_test = bounds[0] * np.ones((D, N_test)) + (bounds[1] - bounds[0]) * np.random.rand(D, N_test)
    N_train = X_train.shape[1]
    N = N_train + N_test
    Xs = np.concatenate((X_train, X_test), axis = 1)
    
    ## Compute K (the matrix of pairwise covariances), using exp. cov. kernel
    K = np.zeros([N, N])
    for i  in range(N):
        dummy = Xs[:,i:(i+1)] - Xs # for (partial) vectorisation
        K[:,i] = np.exp(-np.sum(dummy ** 2, 0) ** 0.5)
    
    k = K[: N_train, : N_train]
    k_test = K[ N_train :, : N_train]
    
    ## Calculating interpolated values
    k_inv = np.linalg.inv(k)# + (10 ** -15) * np.identity(N_train))
    mean_f_x_test = k_test @ k_inv @ Y_train.T
    
    condk = linalg.norm(k) * linalg.norm(k_inv)
    
    ## Error calulations
    Y_test = func(X_test, D)   
    MeanRelError = np.mean(np.abs((mean_f_x_test - Y_test)/Y_test))
    RMSE = np.sqrt(np.sum(np.power(mean_f_x_test - Y_test, 2)) / N_test)
    MaxError = np.max(np.abs(mean_f_x_test - Y_test))
    return(MeanRelError, RMSE, MaxError, condk, X_test, mean_f_x_test, X_train, Y_test)