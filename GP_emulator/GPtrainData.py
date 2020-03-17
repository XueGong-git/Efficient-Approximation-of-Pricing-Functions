import numpy as np
from modelfunc import modelfunc
from GPsparsetrain2 import sparsetrain2
import time

D = 5
Nd = 2**5

tS = time.time()
X_train, Y_train, N_train = sparsetrain2(D, modelfunc, Nd)
print("Train time:", time.time()-tS, "seconds")

np.save('X_train1d32.npy', X_train)
np.save('Y_train1d32.npy', Y_train)