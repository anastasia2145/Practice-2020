import numpy as np
import xgboost as xgb

matrix = np.load("data/test_1000.npy")
print("Matrix loaded")
print(matrix.shape)

X_train, y_train = matrix[:, :-1], matrix[:, -1]
dtrain = xgb.DMatrix(X_train, label=y_train)

param = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',            
            'colsample_bytree': 0.001,
            'learning_rate': 0.1,
            'n_estimators': 12
        }
        
        
xg_reg = xgb.XGBRegressor(**param)

print("Before train")
xg_reg.fit(X_train, y_train, 
           eval_set=[(X_train, y_train)],
           eval_metric='rmse',
           verbose=True)

print("Before save")
xg_reg.save_model('xgb_model')
