import numpy as np
import xgboost as xgb

#matrix = np.load("test_1000.npy")
#print("Matrix loaded")
#print(matrix.shape)

#dtrain = xgb.DMatrix(matrix[:, :-1], label=matrix[:, -1])
#dtrain.save_binary('train.buffer')

dtrain = xgb.DMatrix('data/train_1000.svm.txt')
dtest = xgb.DMatrix('data/train_556.svm.txt')
print("dtrain loaded")

param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',            
            'colsample_bytree': 0.05,
            #'colsample_bylevel': 0.6,
            #'colsample_bynode': 0.6,
            'learning_rate': 0.01
            #'lambda': 2
            
        }
        
print("Before train")       
eval_list = [(dtrain, 'train'), (dtest, 'eval')]
bst = xgb.train(param, dtrain, 700, eval_list, verbose_eval=1)
#bst = xgb.train(param, dtrain, 100, eval_list, verbose_eval=1, xgb_model='xgb_model_new')
print("Before save")
bst.save_model('xgb_model_new_05')
