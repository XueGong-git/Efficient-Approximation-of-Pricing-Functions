import numpy as np
import L2_projection as proj
import total_product as tp
import mlmc_pricing as ml
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import time
np.random.seed(1)

# Fix parameters for MLMC
M = 7
eps = 10**(-3)
extrap = 1

# Parameters for L2 projection
dmu = 3  # Dimensions of mu
d = 4 + dmu  # Total dimensions
nfull = int(12*10**3)  # Number of training samples
nhalf = int(nfull/2)
r = 0.5  # Used to tune convergence rate in n
max_basis_full = int((1 - np.log(2))*nfull/((2 + 2*r)*np.log(nfull)))
max_basis_half = int((1 - np.log(2))*nhalf/((2 + 2*r)*np.log(nhalf)))
deg_full = tp.cost_basis(d, max_basis_full)
deg_half = tp.cost_basis(d, max_basis_half)
n_test = 10**3  # Number of test points
basis_full = tp.basis(deg_full, d)[:max_basis_full, :]  # Choose total product basis
basis_half = tp.basis(deg_half, d)[:max_basis_half, :]  # Choose total product basis

# Fix parameters for model
# Intervals of parameters
I_alpha = [1, 2]
I_T = [0,10]
I_sigma = [0,1]
I_r0 = [0.01,0.05]
I_mu = [0.01,0.05]

change_I = lambda x, newI: (newI[1] - newI[0])*(x + 1)/2 + newI[0]  # Changes interval of chebyshev points

# Define pricing function approximation via mlmc
P = lambda x:  ml.mlmc(M, eps, extrap, change_I(x[0], I_alpha) ,\
                       change_I(x[4:], I_mu), change_I(x[1], I_T), \
                           change_I(x[2], I_sigma), change_I(x[3], I_r0))[0]

# Generate training data
print('Generating training data')
X = proj.Chebyshev_sample((d, nfull)) # Parameter values

# First half
# np.save('/home/s1541634/Documents/Semester 2 Group Project/Python/Data/X_pricing_12k_eps4.npy', X)
prices = np.zeros((1, nfull))  # Prices obtained recursively
time_mlmc_half = 0
for i in range(nhalf):
    print(i)
    t0 = time.perf_counter()
    prices[0, i] = P(X[:,i])
    t1 = time.perf_counter()
    time_mlmc_half += t1 - t0

# Second half
time_mlmc_full = time_mlmc_half
for i in np.arange(nhalf, nfull):
    print(i)
    t0 = time.perf_counter()
    prices[0, i] = P(X[:,i])
    t1 = time.perf_counter()
    time_mlmc_full += t1 - t0
# np.save('/home/s1541634/Documents/Semester 2 Group Project/Python/Data/prices_12k_eps4.npy', prices)       


def L2proj_pricing(x, y, d, n, basis):
    w = lambda x: np.pi**(d)*np.prod(np.sqrt(1 - x**2), axis = 0) / (2**d)
    L = proj.Legendre(x, np.max(basis), basis)  # Evaluate basis functions at training data  
    (coefficients, G) = proj.L2_proj(y, w(x), L)  # Compute coefficients of approximation among basis functions
    f_hat = lambda x: coefficients.reshape(1,coefficients.size)@proj.Legendre(x, np.max(basis), basis) # Evaluates optimal projection 
    return (coefficients, f_hat, G)

# Generate test data
print('Generating test data')
X_test = np.random.uniform(-1, 1, size = (d, n_test))
# np.save('/home/s1541634/Documents/Semester 2 Group Project/Python/Data/X_pricing_test_12k_eps4.npy', X_test)
prices_test = np.zeros((1, n_test))  # Test prices obtained recursively
time_mlmc_test = 0
for i in range(n_test):
    print(i)
    t0 = time.perf_counter()
    prices_test[0, i] = P(X_test[:,i])
    t1 = time.perf_counter()
    time_mlmc_test += t1 - t0
# np.save('/home/s1541634/Documents/Semester 2 Group Project/Python/Data/prices_test_12k_eps4.npy', prices_test)   

# Fit (half) surrogate model
print('Fitting model')
t0 = time.perf_counter()
(SM_half, P_hat_half, G_half) = L2proj_pricing(X[:, :nhalf], prices[0,:nhalf], d, nhalf, basis_half)
t1 = time.perf_counter()
time_train_sm_half = t1 - t0

# Evaluate SM at test points
t0 = time.perf_counter()
SM_test_half = P_hat_half(X_test)
t1=time.perf_counter()
time_sm_half = t1 - t0

# Error analysis of (half) surrogate model
L2_err_half = np.sqrt(np.sum((SM_test_half - prices_test)**2)/n_test)
max_abs_err_half = np.max(np.abs(SM_test_half - prices_test))
Rel_err_half = np.sqrt(np.sum((SM_test_half - prices_test)**2/(prices_test**2))/n_test)
max_Rel_err_half = np.max(np.abs(SM_test_half - prices_test)/prices_test)

# Fit (full) surrogate model
print('Fitting model')
t0 = time.perf_counter()
(SM_full, P_hat_full, G_full) = L2proj_pricing(X[:,:nfull], prices[:,:nfull], d, nfull, basis_full)
t1 = time.perf_counter()
time_train_sm_full = t1 - t0

# Evaluate SM at test points
t0 = time.perf_counter()
SM_test_full = P_hat_full(X_test)
t1=time.perf_counter()
time_sm_full = t1 - t0

# Error analysis of (full) surrogate model
L2_err_full = np.sqrt(np.sum((SM_test_full - prices_test)**2)/n_test)
max_abs_err_full = np.max(np.abs(SM_test_full - prices_test))
Rel_err_full = np.sqrt(np.sum((SM_test_full - prices_test)**2/(prices_test**2))/n_test)
max_Rel_err_full = np.max(np.abs(SM_test_full - prices_test)/prices_test)

print('n:  ', nhalf)
print('Offline/training time (minutes): ', (time_mlmc_half + time_train_sm_half)/60)
print('MLMC time per sample:  ', (time_mlmc_half + time_mlmc_test)/(nhalf + n_test))
print('SM time per test sample:  ', time_sm_half/n_test)
print('L2 error:  ', L2_err_half)
print('Mean relative error (%):  ', Rel_err_half*100)
print('Condition no. of G:  ', np.linalg.cond(G_half))

print('\n')

print('n:  ', nfull)
print('Offline/training time (minutes): ', (time_mlmc_full + time_train_sm_full)/60)
print('MLMC time per sample:  ', (time_mlmc_full + time_mlmc_test)/(nfull + n_test))
print('SM time per test sample:  ', time_sm_full/n_test)
print('L2 error:  ', L2_err_full)
print('Mean relative error (%):  ', Rel_err_full*100)
print('Condition no. of G:  ', np.linalg.cond(G_full))





# Explore Convergence Properties
npoints = 120 # Number of points to evaluate convergence
basis_td = tp.basis(deg_full, d)[:max_basis_full, :]  # Total degree basis
basis_tp = tp.total_product(deg_full, d)[:max_basis_full,:]  # Total product basis

# Compute L2 error at test points for random samples (without repetition) of size i*10**3, i=1,..,npoints
num_average = 20  # Average over 20 random samples of points
err_td = np.zeros(npoints)
err_tp = np.zeros(npoints)
ns = np.arange(1,npoints + 1)*10**2
mb = np.zeros(npoints)
for i in np.arange(1, npoints + 1):
    n_temp = int(i*(nfull / npoints))
    mb[int(i-1)] = int((1 - np.log(2))*n_temp/((2 + 2*r)*np.log(n_temp)))
    
    basis_temp_td = basis_td[:int(mb[int(i-1)]),:]
    basis_temp_tp = basis_tp[:int(mb[int(i-1)]),:]
    for p in range(num_average):       
        train_points = np.arange(0, nfull)
        np.random.shuffle(train_points)
        train_points = train_points[:n_temp]
        
        X_temp = X[:,train_points]
        prices_temp = prices[0,train_points]
        
        (fit_temp_td, P_temp, G_temp) = L2proj_pricing(X_temp, prices_temp, d, n_temp, basis_temp_td)
        SM_temp = P_temp(X_test)
        err_td[int(i - 1)] += np.sqrt(np.sum((SM_temp - prices_test)**2)/n_test)/num_average
        
        (fit_temp_tp, P_temp, G_temp) = L2proj_pricing(X_temp, prices_temp, d, n_temp, basis_temp_tp)
        SM_temp = P_temp(X_test)
        err_tp[int(i - 1)] += np.sqrt(np.sum((SM_temp - prices_test)**2)/n_test)/num_average

fig, ax = plt.subplots(1, 1, figsize = (10,5))
ax.loglog(ns, err_td, 'r-', label = 'Total Degree')
ax.loglog(ns, err_tp,  label = 'Total Product')
ax.set_xlabel(r'$n$', fontsize = 16)
ax.set_ylabel(r'$||P - \hat P||_{L^2}$', fontsize = 16)
ax.legend(loc = 'upper right', fontsize = 14)
ax.tick_params(labelsize = 12)

fig2, ax2 = plt.subplots(1,1, figsize = (10,5))
em2_td = [np.sqrt(np.sum(fit_temp_td[int(mb[i]):]**2)) for i in range(npoints)]
em2_tp = [np.sqrt(np.sum(fit_temp_tp[int(mb[i]):]**2)) for i in range(npoints)]
ax2.loglog(ns, em2_td, 'r-', label = 'Total Degree')
ax2.loglog(ns, em2_tp,  label = 'Total Product')
ax2.set_xlabel(r'$n$', fontsize = 16)
ax2.set_ylabel(r'2-norm of unused coefficients', fontsize = 16)
ax2.tick_params(labelsize = 12)
ax2.legend(loc = 'upper right', fontsize = 14)
