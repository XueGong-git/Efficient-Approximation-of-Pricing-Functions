from sg_points import sparseGridPosList
import numpy as np

### Generate sparse grid and evaluate P at those points using MLMC

def sparsetrain2(D, func, Nd):
    L = int(np.log2(Nd)) # max level, L, is log_2(Nd)
    X_train = sparseGridPosList(D, L)
    Y_train = func(X_train)
    return(X_train, Y_train, X_train.shape[1])