import numpy as np
import xgboost as xgb

train = np.load("data/train_556_conv.npy")
#val = np.load("data/train_556.npy")
print("Matrix loaded")

X_train, y_train = train[:, :-1], train[:, -1]
#X_val, y_val = val[:, :-1], val[:, -1]
print(y_train[:20])


param = {
            'objective': 'multi:softmax',
            'eval_metric': ['merror'],
            'colsample_bytree': 0.6,
            'colsample_bylevel': 0.6,
            'colsample_bynode': 0.6,
            'learning_rate': 0.01,
            'lambda': 2,
            'n_estimators' : 100,
            'num_class': 8,
            'random_state': 21
        }

clf = xgb.XGBClassifier(**param)

print("Before train")
clf.fit(X_train, y_train,
        eval_set=[(X_train, y_train)],
        eval_metric=['merror'],
        verbose=True)     

print("Before save")
clf.save_model('xgb_model_clf')
