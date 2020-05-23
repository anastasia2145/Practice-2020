import numpy as np
import xgboost as xgb

from matplotlib.legend_handler import HandlerLine2D
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import matplotlib.pyplot as plt

dtrain = xgb.DMatrix('data/train_1000.svm.txt')
dtest = xgb.DMatrix('data/train_556.svm.txt')

colsample_bytrees = np.linspace(0, 0.8, 5, endpoint=True)
train_results = []
test_results = []
for colsample_bytree in colsample_bytrees:
    param = {'objective': 'reg:squarederror', 'colsample_bytree': colsample_bytree}
    reg = xgb.train(param, dtrain, 300)
    train_pred = reg.predict(dtrain)
    rmse = mean_squared_error(dtrain.get_label(), train_pred, squared=False)
    train_results.append(rmse)
    y_pred = reg.predict(dtest)
    rmse = mean_squared_error(dtest.get_label(), y_pred, squared=False)
    test_results.append(rmse)
                                        
                                            
line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train RMSE')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test RMSE')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('RMSE score')
plt.xlabel('colsample_bytree')
plt.savefig('plots/colsample_bytree_reg.png')
