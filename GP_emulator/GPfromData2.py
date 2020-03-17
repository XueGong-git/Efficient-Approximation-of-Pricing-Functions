import numpy as np
from modelfunc import modelfunc
from GPoffline import offline
### Implement timed offline and online stages of GPR with sparse grid of
### training points given pre-generated training data which is loaded here.


X_train = np.load('X_train3d32.npy')
Y_train = np.load('Y_train3d32.npy')

N_train = Y_train.shape[1]

N_test = 200
D = 7

condk, X_test, X_train, Y_test, Y_train = offline(D, modelfunc, X_train, Y_train, N_test, (0,1))

runfile('/home/s1511699/Documents/Coding/Python code/GP emulators//Git/GPonline.py', wdir='/home/s1511699/Documents/Coding/Python code/GP emulators/Git')