import time
import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt

mic2class = {0.25: 0, 1.0 : 1, 2.0: 2, 0.5: 3, 4.0: 4, 8.0: 5, 16.0: 6}


# train = np.load("data/matrix_test_norm.npy")
# test = np.load("data/test.npy")

matrix = np.load("data/norm_matrix.npy")
permutation = np.load("data/permutation.npy")
matrix = matrix[permutation]
train = matrix[:1200]
test = matrix[1200:]

x_train = train[:, :-1]
y_train = train[:, -1]

x_test = test[:, :-1]
y_test = test[:, -1]

n_estimators = [1, 2, 4, 8, 16, 32, 64, 100]
train_results = []
test_results = []

for estimator in n_estimators:
    start = time.time()
    rf = RandomForestRegressor(n_estimators=estimator, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    
    rmse = mean_squared_error(y_train, train_pred, squared=False)
    train_results.append(rmse)

    y_pred = rf.predict(x_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    test_results.append(rmse)
    print("Finished {} in {} minute, train_rmse: {}   test_rmse: {}".format(estimator, (time.time()-start)/60, train_results[-1], test_results[-1] ))
    

line1, = plt.plot(n_estimators, train_results, 'b', label='Train RMSE')
line2, = plt.plot(n_estimators, test_results, 'r', label='Test RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('RMSE score')
plt.xlabel('n_estimators')
# plt.show()
plt.savefig('plots/n_estimators_reg.png')
