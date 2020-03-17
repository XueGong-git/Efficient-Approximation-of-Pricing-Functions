### Implement GP regression, given D, training data, Nd, and bounds, to obtain 
### interpolated values for test points sampled uniformly at random, then 
### return the error for convergence tests. Uses specific cov. kernel.

import numpy as np
import numpy.linalg as linalg
import time

def offline(D, func, X_train, Y_train, N_test, bounds):
    tOverheadStart = time.time()
    
    X_test = bounds[0] * np.ones((D, N_test)) + (bounds[1] - bounds[0]) * np.random.rand(D, N_test)
    N_train = X_train.shape[1]
    N = N_train + N_test
    Xs = np.concatenate((X_train, X_test), axis = 1)
    
    ## Compute K (the matrix of pairwise covariances), using choice of k(u,u')
    K = np.zeros([N, N])
    for i  in range(N):
        dummy = Xs[:, i : (i + 1)] - Xs # for (partial) vectorisation
        #K[:, i] = np.exp(-np.sum(dummy ** 2, 0) ** 0.5) # exp. Matern
        K[:, i] = np.exp(-np.sum(np.abs(dummy), 0)) # separable exp. Matern
        #K[:, i] = np.exp(-np.sum(dummy ** 2, 0)) # Gaussian
        #K[:,i] = (1+np.sqrt(3) * np.sum(np.abs(dummy), 0)) * (np.exp(-np.sqrt(3) * np.sum(np.abs(dummy), 0))) # separable Matern with nu=1.5
    
    k = K[: N_train, : N_train]
    k_test = K[ N_train :, : N_train]
    
    ## Calculating interpolated values
    tInvStart = time.time()
    k_inv = np.linalg.inv(k)# + (10 ** -4) * np.identity(N_train)
    tInv = time.time() - tInvStart
    print("Inverting k:", tInv, "seconds")
    
    k_invY_train = k_inv @ Y_train.T
    tOverhead = time.time() - tOverheadStart
    print("Overhead:", tOverhead, "seconds")
    
    condk = linalg.norm(k) * linalg.norm(k_inv)
    
    ## Error calulations
    tTestMLMCStart = time.time()
    Y_test = func(X_test)
    tTestMLMC = time.time() - tTestMLMCStart
    print("Test MLMC:", tTestMLMC, "seconds")
    
    np.save('k_invY_train.npy', k_invY_train)
    np.save('k_test.npy', k_test)
    
    return(condk, X_test, X_train, Y_test, Y_train)