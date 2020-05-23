import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, precision_recall_fscore_support, accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score

class2mic = {0: 0.25, 1: 1.0, 2: 2.0, 3: 0.5, 4: 4.0, 5: 8.0, 6: 16.0}
mic2class = {0.25: 0, 1.0 : 1, 2.0: 2, 0.5: 3, 4.0: 4, 8.0: 5, 16.0: 6}
bst = xgb.Booster()
#bst.load_model('xgb_model_clf')
bst.load_model('xgb_model_clf_accuracy_75_weighted')

matrix = np.load("data/matrix_test_norm.npy")
y_test = matrix[:, -1]
#y_test = [0 if i <= 4 else 1 for i in y_test]
y_test_conv = np.array([mic2class[i] for i in y_test])

dtest= xgb.DMatrix(matrix[:, :-1])
y_pred_conv = bst.predict(dtest)
#y_pred_conv = [int(i) for i in y_pred_conv]
#print(pred[:5])
#y_pred_conv = [np.argmax(i) for i in pred]
print(type(y_test_conv))
print(type(y_pred_conv))
print(y_pred_conv.shape)
y_pred = [class2mic[i] for i in y_pred_conv]

print(precision_recall_fscore_support(y_test_conv, y_pred_conv))                                                    
print(accuracy_score(y_test_conv, y_pred_conv))
#y_pred = [class2mic[i] for i in y_pred]
print(mean_squared_error(y_test, y_pred, squared=False))
y_test_conv = y_test_conv.reshape((-1,1))
y_pred_conv = y_pred_conv.reshape((-1,1))

#false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_conv, y_pred_conv)
#roc_auc = auc(false_positive_rate, true_positive_rate)
#roc_auc = roc_auc_score(y_test_conv, y_pred_conv, multi_class='ovo')
#print("AUC_ROC: ", roc_auc)


print(y_test[:20])
print(y_pred[:20])
