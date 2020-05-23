import numpy as np
import xgboost as xgb


dtrain = xgb.DMatrix('data/train_1000_conv.svm.txt')
dval = xgb.DMatrix('data/train_556_conv.svm.txt')
y_train = np.load("data/y_train.npy")
mic2weights = {0.25: 0.290165,
                0.5: 0.41452143,
                1.0:  0.1450825,
                2.0: 0.0098361,
                4.0: 0.03495964,
                8.0: 0.00871366,
                16.0: 0.09672167}   
weights = [mic2weights[i] for i in y_train]
dtrain.set_weight(weights[:1000])
dval.set_weight(weights[1000:])

#train_label = dtrain.get_label()
#val_label = dval.get_label()

#train_label = [0 if i <= 4 else 1 for i in train_label]
#val_label = [0 if i <= 4 else 1 for i in val_label] 

#dtrain.set_label(train_label)
#dval.set_label(val_label)
print("dtrain loaded")

param = {
            'objective': 'multi:softmax',
            'eval_metric': ['merror'],
            'colsample_bytree': 0.8,
            #'colsample_bylevel': 0.2,
            #'colsample_bynode': 0.3,
            'learning_rate': 0.05,
            'max_depth': 4,
            #'lambda': 6,
            'num_class': 8,
            'random_state': 21
        }

print("Before train")
#eval_list = [(dtrain, 'train'), (dval, 'eval')]
#bst = xgb.train(param, dtrain, 20, eval_list, verbose_eval=1)
eval_list = [(dval, 'train')]
bst = xgb.train(param, dval, 20, eval_list, verbose_eval=1, xgb_model='xgb_model_clf')
print("Before save")
bst.save_model('xgb_model_clf')
