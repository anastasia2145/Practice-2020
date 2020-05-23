import pickle
import numpy as np

from sklearn.metrics import mean_squared_error
from collections import Counter 

test = np.load("data/matrix_test_norm.npy")

X_test, y_test = test[:, :-1], test[:,-1]

with open('models/lasso_lars.pickle', 'rb') as f:
    reg = pickle.load(f)

y_pred = reg.predict(X_test)
print("RMSE: {}".format(mean_squared_error(y_test, y_pred, squared=False)))

print(y_pred[:20])
print(y_test[:20])

print(Counter(reg.coef_))