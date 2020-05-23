import numpy as np
import pickle
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc

class2mic = {0: 0.25, 1: 1.0, 2: 2.0, 3: 0.5, 4: 4.0, 5: 8.0, 6: 16.0}
mic2class = {0.25: 0, 1.0 : 1, 2.0: 2, 0.5: 3, 4.0: 4, 8.0: 5, 16.0: 6}

with open('models/randomforect_bin_clf.pickle', 'rb') as f:
    clf = pickle.load(f)
    
test = np.load("data/matrix_test_norm.npy")
y_test = test[:, -1]
#y_test_conv = [mic2class[i] for i in y_test]
y_test_conv = [0 if i <=4 else 1 for i in y_test]

y_pred_conv = clf.predict(test[:,:-1])
#y_pred = [class2mic[i] for i in y_pred_conv]

print(precision_recall_fscore_support(y_test_conv, y_pred_conv))
print("Accuracy: {}".format(accuracy_score(y_test_conv, y_pred_conv)))
print("RMSE: {}".format(mean_squared_error(y_test_conv, y_pred_conv, squared=False)))

#false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test_conv, y_pred_conv)
#roc_auc = auc(false_positive_rate, true_positive_rate)
#print("AUC: {}".format(roc_auc))


for i in range(20):
    print(y_test[i], y_test_conv[i], y_pred_conv[i])

#print(y_test[:20])
#print(y_pred[:20])
