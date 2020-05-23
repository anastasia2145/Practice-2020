import numpy as np
import xgboost as xgb


dtrain = xgb.DMatrix('data/train_1000_conv.svm.txt')
dval = xgb.DMatrix('data/train_556_conv.svm.txt')
y_train = np.load("y_train.npy")
mic2weights = {0.25: 0.290165,
                0.5: 0.41452143,
                1.0:  0.1450825,
                2.0: 0.0098361,
                4.0: 0.03495964,
                8.0: 0.00871366,
                16.0: 0.09672167}   
#weights = [mic2weights[i] for i in y_train]
#dtrain.set_weight(weights[:1000])
#dval.set_weight(weights[1000:])
print("dtrain loaded")

param = {
            'objective': 'multi:softmax',
            'eval_metric': ['merror'],
            'colsample_bytree': 0.1,
            'colsample_bylevel': 0.1,
            'colsample_bynode': 0.1,
            'learning_rate': 0.01,
            'max_depth': 4,
            'lambda': 6,
            'num_class': 8,
            'random_state': 21
        }

print("Before train")
eval_list = [(dtrain, 'train'), (dval, 'eval')]
bst = xgb.train(param, dtrain, 100, eval_list, verbose_eval=1)
#eval_list = [(dtrain, 'train')]
#bst = xgb.train(param, dtrain, 200, eval_list, verbose_eval=1, xgb_model='xgb_model_cl_22')
print("Before save")
bst.save_model('xgb_model_clf')
