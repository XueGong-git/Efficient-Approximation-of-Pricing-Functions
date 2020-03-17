# Total product basis
import numpy as np


# Create function to take element lambda = (lambda_1, ...., lambda_d) in total degree basis and return the indices
# lambda_hat satisfying 2^{lambda} - 1 <= lambda_hat <= 2(2^{lambda} - 1).
def map_lda(lda):
    n_map = np.prod(2**lda)
    n_map0 = 2**lda[0]
    beta = np.zeros((n_map, lda.size))
    count = 0
    if lda.size == 1:
        if lda[0]>0:
            return np.arange(2**(lda[0]) - 1, 2**(lda[0]+1) -1).reshape((n_map, 1))
        else:
            return np.array([0]).reshape((1,1))
    else:
        if lda[0]>0:
            for a0 in np.arange(2**(lda[0]) - 1, 2**(lda[0] + 1) - 1):
                beta[count:count+int(n_map/n_map0),:] = \
                    np.concatenate((a0*np.ones((int(n_map/n_map0),1)), map_lda(lda[1:])), axis = 1)
                count += int(n_map/n_map0)
        else:
            beta[:,:] = np.concatenate((np.zeros((n_map,1)), map_lda(lda[1:])), axis = 1)

    return beta.astype(int)

# Create function to compute total degree basis
def basis(deg, d):
    basis = np.zeros((1, d))
    B0 = np.eye(d)
    basis = np.concatenate((basis,B0), axis = 0)
    test = True
    while test == True:
        B1 = np.zeros((B0.shape[0]*d, d))
        count = 0
        for b in B0:
            B1[count:count + d] = b + np.eye(d)
            count += d
        B1 = np.unique(B1, axis = 0)
        B1 = B1[[np.sum(B1[j,:]) <= deg for j in range(B1.shape[0])],:]
        if B1.size == 0:
            test = False
        B0 = B1
        basis = np.concatenate((basis,B0), axis = 0)   
    return basis.astype(int)
        

# Create function to compute total product basis
def total_product(L, d):
    lda = basis(L, d)
    beta = map_lda(lda[0,:])
    for a in lda[1:,:]:
        beta = np.concatenate((beta, map_lda(a)), axis = 0)
    return beta.astype(int)

# Function to compute the smallest basis which contains max_basis elements, for total product basis
def cost_tp(d, max_basis):
    for L in range(100):
        if total_product(L, d).shape[0] > max_basis:
            return L  
    return L

# Function to compute the smallest basis which contains max_basis elements, for total degree basis
def cost_basis(d, max_basis):
    for L in range(100):
        if basis(L, d).shape[0] > max_basis:
            return L 
    return L
