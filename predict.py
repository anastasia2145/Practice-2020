import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, accuracy_score

class2mic = {0: 0.25, 1: 1.0, 2: 2.0, 3: 0.5, 4: 4.0, 5: 8.0, 6: 16.0}
mic2class = {0.25: 0, 1.0 : 1, 2.0: 2, 0.5: 3, 4.0: 4, 8.0: 5, 16.0: 6}
bst = xgb.Booster()
bst.load_model('xgb_model_lambda_part_2')

#print(bst.get_xgb_params())

matrix = np.load("data/matrix_test_norm.npy")
y_test = matrix[:, -1]
#y_test_conv = [mic2class[i] for i in y_test]

dtest= xgb.DMatrix(matrix[:, :-1])
y_pred = bst.predict(dtest)

#print(precision_recall_fscore_support(y_test_conv, y_pred))                                                             
#print(accuracy_score(y_test_conv, y_pred))
#y_pred = [class2mic[i] for i in y_pred]
print(mean_squared_error(y_test, y_pred, squared=False))
print(y_test[:20])
print(y_pred[:20]) 

#print('Model Report %r' % (classification_report(y_test, y_pred)))
