### Produce convergence tests of GP interpolation given D, bounds, func, 
### (X_train, Y_train).

# Note: covariance kernel is specified in GPreg for the purpose of vectorising.

import numpy as np
import matplotlib.pyplot as plt
from GPreg import reg
from GPtrain import train
from GPsparsetrain import sparsetrain
from GPsparsetrain2 import sparsetrain2
import time
import numpy.linalg as linalg

tStart = time.time()


D = 2

bounds = (-1, 1)

func = lambda X, D: np.sin(np.sum(X, axis = 0) * 2 * np.pi / D) # function to be interpolated

# cov = lambda x_1, x_2: np.exp(-0.5 * np.linalg.norm((x_1 - x_2), 2)) # exponential covariance kernel

N_test = 100



## Generate range, ns, of numbers of training points per dimension, Nd.

#oneD_max = 5000
nInc = 5
#mD_inc = np.ceil(np.power(oneD_max, 1 / D) / nInc)
#ns = np.arange(mD_inc, (nInc + 1) * mD_inc, mD_inc)
#ns = np.geomspace(3, np.power(oneD_max, 1 / D), nInc, endpoint = True) # log spaced values for Nd from 3 to max affordable

ns = np.power(2, np.arange(nInc)) * (2 ** 2)
#ns = (2,4,8,16,32,64,128)

print("ns =", ns)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~Computation~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
tCompStart = time.time()

MeanRelError = np.zeros(nInc)
RMSE = np.zeros(nInc)
MaxError = np.zeros(nInc)

for p in range(nInc):
    ## Generate training points.
    Nd = int(ns[p])
    X_train, Y_train, N_train = sparsetrain2(D, func, Nd, bounds)
    #X_train, Y_train, N_train = train(D, func, Nd, bounds)
    
    ## Compute errors for a range of numbers of training points.
    MeanRelError[p], RMSE[p], MaxError[p], condk, X_test, mean_f_x_test, X_train, Y_test = reg(D, func, X_train, Y_train, N_test, bounds)
    print("Point", p + 1,"done")
    
print("Time for just computation:", time.time() - tCompStart, "seconds")
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    
# Log-log plots of errors

fig, axs = plt.subplots(1, 3, figsize=(30, 10))
axs = axs.ravel()
plt.rcParams.update({'font.size': 17})

axs[0].loglog(ns**D, MeanRelError, 'x', label='Mean relative error')
axs[0].loglog(ns**D, .1e1 * (ns**D)**-2., ':', label='n^(-2)')
axs[0].legend() # turn on legend
axs[0].set_title('D = '+str(D))

axs[1].loglog(ns**D, RMSE, 'x', label='RMSE')
axs[1].loglog(ns**D, 1e-0 * (ns**D)**-2., ':', label='n^(-2)')
axs[1].legend() # turn on legend
axs[1].set_title('D = '+str(D))

axs[2].loglog(ns**D, MaxError, 'x', label='Max Error')
axs[2].loglog(ns**D, 1e-0 * (ns**D)**-2., ':', label='n^(-2)')
axs[2].legend() # turn on legend
axs[2].set_title('D = '+str(D))

print("Time taken:", time.time() - tStart, "seconds")


fig2, axs2 = plt.subplots(1, 1, figsize=(10, 10))
axs2.plot(X_train[0,:], X_train[1,:], 'x')
plt.xlim(bounds)
plt.ylim(bounds)