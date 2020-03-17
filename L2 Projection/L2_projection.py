# Orthogonal Projection Functions

import numpy as np
from scipy.special import binom
from scipy.special import legendre

############################## Algorithm #########################################################################
# Define norm || v ||_n = (0.5 sum wi*|v(xi)|^2)^0.5 and corresponding inner
# product
# Compute the inner product of functions v, w with values v(xi) = ai and 
# w(xi) = bi, ip_n_vec allows b to be a matrix, computes inner product of v 
# with functions defining each row of b. 
G_vec = lambda L, w:  np.matmul(w*L, L.transpose())/w.size # Vector output
d_vec = lambda L, y, w:  (w*L) @ (y.reshape((y.size,1)))/w.size   # Scalar output


# Define function for L2 projection
def L2_proj(y, w, L):
    """
    y <- n-dimensional vector of observations.
    w <- vector of weights
    L <- m x n matrix with rows L[i,j] = L_i(x_j), containing basis function L_i evaluated at the point x_j 
         corresponding to observation y[j] j=0,..,n-1
    """
    # Generate matrix G and vector d:
    G = G_vec(L, w)
    d = d_vec(L, y, w)
    
    # Solve linear system G*v = d to obtain coefficients v
    coefs = np.linalg.solve(G, d)
    return coefs, G

##################################################################################################################
    
################# Legendre Polynomials ###########################################################################
    
# Create function to sample from Chebyshev measure dp = (1-x^2)^(-1/2)/pi *dx on [-1, 1] using ITS Since the 
# Chebyshev measure has c.d.f arcsin(x)/pi + 1/2, we sample values from its inverse -cos(pi*u) where 
# u ~ Uniform[0,1].
Chebyshev_sample = lambda shape: -np.cos(np.pi*np.random.uniform(0,1,shape))

# Create function to take a (deg + 1) x (n x d) matrix M, divides M into d matrices of size (deg + 1) x n
# by partitioning after every n columns of M, and returns the matrix L obtained by multiplying componentwise
# rows such that the total degree is <= deg

def Legendre_univ(x, deg):
    """
    Returns all Legendre polynomials of degree 0<=d<=deg evaluated at the points x in [-1, 1]. 
    """
    L = np.zeros((deg + 1, x.size))
    for i in range(deg + 1):
        if i==0:
            L[i,:] += 1
        elif i==1:
            L[i,:] = x
        else:
            L[i,:] = ( (2*i-1) * x * L[i - 1,:] - (i-1) * L[i-2,:]) / i
            
    normalize_constant = np.sqrt(2*np.arange(0,deg + 1) + 1).reshape((deg + 1,1))
    L = normalize_constant*L
    return L

# Create product of Legendre polynomials evaluated at samples x according to a specified index.
def Legendre(x, deg, Basis):
    """
    x -> d x n matrix of n samples x[:,i] \i \R^d
    deg -> Maximum degree of polynomials n the basis
    Returns d - dimensional basis functions wth total degree <= deg evaluated at the points in x
    """
    # Compute number of dimensions and samples from x

    d = x.shape[0]  # No. dimensions
    n = x.shape[1]  # No. samples
    m = Basis.shape[0]  # No. basis functions
    # Generate univariate Lebesgue polynimials in first dimension via recursion
    L_combine = []
    for i in range(d):
        L_combine.append(Legendre_univ(x[i, :], deg))
        
    L = np.zeros((m, n))
    count = 0
    for b in Basis:
        L[count,:] = np.prod([L_combine[i][b[i],:] for i in range(d)], axis = 0)
        count += 1
    
    return L
