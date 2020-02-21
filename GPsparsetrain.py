import numpy as np
import numpy.linalg as linalg

def sparsetrain2(D, func, Nd, bounds):
    L = int(np.log2(Nd))
    
    for i in range(0, L + 1):
        for j in range(0, L - i + 1):
            x = bounds[0] + (np.linspace(0, 1, 2**i, endpoint = False) + (1 / (2 * (2 ** i)))) * (bounds[1] - bounds[0])
            
            y = bounds[0] + (np.linspace(0, 1, 2**j, endpoint = False) + (1 / (2 * (2 ** j)))) * (bounds[1] - bounds[0])
            
            c = (x,y)
            submesh = np.array(np.meshgrid(*c))
            sublist = np.reshape(submesh, (D, (2 ** i) * (2 ** j)))
            
            if i + j == 0:
                list_ = sublist.copy() # set list_ to equal first round of points
            else:      
                list_ = np.concatenate((list_, sublist), axis = 1)
                    
    ## Remove points which are too close together
    #list_2 = list_.copy()
    #for l in range(list_2.shape[1]):
    #    if l > list_2.shape[1] - 1:
    #        break
    #    dl = list_2[:, l:l+1] - list_2
    #    norm_dl = linalg.norm(dl, ord=2, axis=0)
    #    norm_dl = norm_dl * np.ones((1, norm_dl.size))
    #    if sum(sum(norm_dl < 2 ** (-1))) > 1:
    #        list_2 = np.delete(list_2, obj=l, axis=1)
                
#    X_train = list_.copy()
#    N_train = list_.shape[1]
    
    Y_train = func(list_, D)# * np.ones((1, list_.shape[1])) # np.ones is just to give Y_train second axis here.
    return(list_, Y_train, list_.shape[1])
