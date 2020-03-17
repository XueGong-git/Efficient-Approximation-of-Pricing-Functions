import numpy as np
import time


#X_train = np.load('X_train3d64.npy')
#Y_train = np.load('Y_train3d64.npy')

#N_train = Y_train.shape[1]

#N_test = 200



k_invY_train = np.load('k_invY_train.npy')
k_test = np.load('k_test.npy')




tOnlineStart = time.time()

mean_f_x_test = k_test @ k_invY_train

tOnline = time.time() - tOnlineStart
print("Online:", tOnline, "seconds")



tErrorsStart = time.time()
MeanRelError = np.mean(np.abs((mean_f_x_test.T - Y_test)/Y_test))
MaxRelError = np.max(np.abs((mean_f_x_test.T - Y_test)/Y_test))
RMSE = np.sqrt(np.sum(np.power(mean_f_x_test.T - Y_test, 2)) / N_test)
MaxError = np.max(np.abs(mean_f_x_test.T - Y_test))
L2Error = np.sqrt((1 / N_test) * np.sum(np.power(np.abs(mean_f_x_test.T - Y_test), 2)))
tErrors = time.time() - tErrorsStart
print("Error calculations:", tErrors, "seconds")