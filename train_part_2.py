import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error

#bst = xgb.Booster()
#bst.load_model('xgb_model_5')


dtrain = xgb.DMatrix('data/train_556.svm.txt')

param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'colsample_bytree': 0.05,
            #'colsample_bylevel': 0.6,
            #'colsample_bynode': 0.6,
            'learning_rate': 0.01
            #'lambda': 2
        }
eval_list = [(dtrain, 'train')]
print("Before train")
bst = xgb.train(param, dtrain, 500, eval_list, verbose_eval=1, xgb_model='xgb_model_new_05')

print("Before save")
bst.save_model('xgb_model_new_05_2')

