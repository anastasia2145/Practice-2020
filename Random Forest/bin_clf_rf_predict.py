import numpy as np
import pickle
from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_fscore_support
from sklearn.metrics import roc_curve, auc

class2mic = {0: 0.25, 1: 1.0, 2: 2.0, 3: 0.5, 4: 4.0, 5: 8.0, 6: 16.0}
mic2class = {0.25: 0, 1.0 : 1, 2.0: 2, 0.5: 3, 4.0: 4, 8.0: 5, 16.0: 6}

with open('models/randomforect_bin_clf.pickle', 'rb') as f:
    clf = pickle.load(f)

test = np.load("data/matrix_test_norm.npy")
y_test_mic = test[:, -1]
y_test= [0 if i <=4 else 1 for i in y_test_mic]

y_pred = clf.predict(test[:,:-1])

print(precision_recall_fscore_support(y_test, y_pred))
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
print("RMSE: ", mean_squared_error(y_test, y_pred, squared=False))
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print("AUC: {}".format(roc_auc))


for i in range(20):
    print(y_test_mic[i], y_test[i], y_pred[i])
