import pysg
import numpy as np
#import math
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

#dim = 7
#levels = 7
def sparseGridPosList(dim, levels):
    sg = pysg.sparseGrid(dim, levels)
    sg.generatePoints()
    n = len(sg.indices)
    positions = np.zeros((dim, n))
    for i in range(n):
        positions[:, i] = sg.gP[tuple(sg.indices[i])].pos
    return(positions)

#positions = sparseGridPosList(dim, levels)
#fig2, axs2 = plt.subplots(1, 1, figsize=(10, 10))
#axs2.plot(positions[0,:], positions[1,:], 'x')

#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(positions[0,:], positions[1,:], positions[2,:])