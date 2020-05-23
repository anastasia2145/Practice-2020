import pickle
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoLars
from sklearn.metrics import mean_squared_error
from matplotlib.legend_handler import HandlerLine2D

train = np.load("data/norm_matrix.npy")
# test = np.load("data/matrix_test_norm.npy")

X_train, y_train = train[:, :-1], train[:,-1]
reg = LassoLars(alpha=0.05)
reg.fit(X_train, y_train)

with open('models/lasso_lars.pickle', 'wb') as f:
    pickle.dump(reg, f)


# X_test, y_test = train[1200:, :-1], train[1200:,-1]

# X_train, y_train = train[:500, :-1], train[:500,-1]
# X_test, y_test = train[500:600, :-1], train[500:600,-1]
# train_results = []
# test_results = []

# alphas = np.linspace(0, 1, 20, endpoint=True)
# alphas = np.arange(20)

# for alpha in alphas:
    # reg = LassoLars(alpha=alpha)
    # reg.fit(X_train, y_train)
    # y_pred = reg.predict(X_train)
    # train_results.append(mean_squared_error(y_train, y_pred, squared=False))
    
    # y_pred = reg.predict(X_test)
    # test_results.append(mean_squared_error(y_test, y_pred, squared=False))
    # print("Alpha {} complited. Train rmse: {}  Test rmse: {} ".format(alpha, \
            # train_results[-1], test_results[-1]))

# line1, = plt.plot(alphas, train_results, 'b', label='Train RMSE')
# line2, = plt.plot(alphas, test_results, 'r', label='Test RMSE')
# plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
# plt.ylabel('RMSE score')
# plt.xlabel('Alpha')
# plt.show()
# plt.savefig('lars_alphas.png')