import numpy as np
import numpy.linalg as linalg

def sparsetraindraft(D, func, Nd, bounds):
    L = int(np.log2(Nd))
    #list_ = np.empty
    count = 0
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for i in range(1, L + 1):
        for j in range(1, L - i + 1):
            x = bounds[0] + (np.linspace(0, 1, 2**i + 1, endpoint = False) + (1 / (2 * (2 ** i + 1)))) * (bounds[1] - bounds[0])
            
            y = bounds[0] + (np.linspace(0, 1, 2**j + 1, endpoint = False) + (1 / (2 * (2 ** j + 1)))) * (bounds[1] - bounds[0])
            
            c = (x,y)
            submesh = np.array(np.meshgrid(*c))
            sublist = np.reshape(submesh, (D, (2 ** i + 1) * (2 ** j + 1)))
            # presublist = sublist.copy()
            count += sublist.shape[1]
            
            if i + j == 2:
                list_ = sublist.copy() # set list_ to equal first round of points
            else:
                nsub = sublist.shape[1] # number of points in subsequent rounds of suggestions
                for z in range(nsub): # check if elements of sublist are close to those already in list
                    if z > sublist.shape[1] - 1: # break if run out of suggestions to check
                        break
                    d = sublist[:, z:z+1] - list_ # displacements of point under consideration from those already stored in list_
                    norm_d = linalg.norm(d, ord=2, axis=0) # distances: l2 norms of displacements above
                    norm_d = norm_d * np.ones((1, norm_d.size))
                    if sum(sum(norm_d < 2 ** (- (L + 1)))) > 0: # check if any distances are too small
                        sublist = np.delete(sublist, obj=z, axis=1) # update list without adding too close points
                if sublist.shape[1] > 0:
                    list_ = np.concatenate((list_, sublist), axis = 1)
                
#    X_train = list_.copy()
#    N_train = list_.shape[1]
    Y_train = func(list_, D) * np.ones((1, list_.shape[1])) # np.ones is just to give Y_train second axis here.
    return(list_, Y_train, list_.shape[1], list_, i, j, L, count)
