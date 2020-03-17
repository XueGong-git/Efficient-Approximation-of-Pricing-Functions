from MLMC_BK_estimation import mlmc
import numpy as np

def modelfunc(X):
    #################################
    ### modelfunc takes list of length 7 sequences as input, representing parameters
    ### of B-K model, implements MLMC using imported function, then returns P.
    ### Inputs are all (0,1), so these are mapped to desired domain.
    #################################
    
    
    # fixed inputs
    M = 2
    eps = 1e-2
    extrap = 0
    
    nTrain = X.shape[1]
    
    # varied inputs
    alpha = X[0, :] + np.ones((1, nTrain))
    T = X[1, :] * 10
    sigma = X[2, :]
    
    r0 = (X[3, :] * 0.04) + (0.01 * np.ones((1, nTrain))) # makes r0 in (0.01, 0.05)
    #r0 = r0.reshape(nTrain)
    
    mu = X[4:7, :] * 0.04 + 0.01 * np.ones((3, nTrain)) # 3D mu
    #mu = X[4, :] * 0.04 + 0.01 * np.ones((1, nTrain)) # scalar mu
    #mu = 0.01 # fixed mu
    
    
    Y = np.zeros((1, nTrain))
    for p in range(nTrain):
        Y[0, p], Nl, mlmc_cost, con = mlmc(M, eps, extrap, alpha[0, p], mu[:, p], T[p], sigma[p], r0[0, p]) # 3d mu
        #Y[0, p], Nl, mlmc_cost, con = mlmc(M, eps, extrap, alpha[0, p], mu[0, p], T[p], sigma[p], r0[0, p]) # scalar mu
        #Y[0, p], Nl, mlmc_cost, con = mlmc(M, eps, extrap, alpha[0, p], mu, T[p], sigma[p], r0[0, p]) # fixed mu
    
    return(Y)