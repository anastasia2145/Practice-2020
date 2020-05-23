import numpy as np
from matplotlib.legend_handler import HandlerLine2D
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib as mpl
import matplotlib.pyplot as plt

mic2class = {0.25: 0, 1.0 : 1, 2.0: 2, 0.5: 3, 4.0: 4, 8.0: 5, 16.0: 6}


# train = np.load("data/matrix_test_norm.npy")
# test = np.load("data/test.npy")

matrix = np.load("data/norm_matrix.npy")
permutation = np.load("data/permutation.npy")
matrix = matrix[permutation]
train = matrix[:1200]
test = matrix[1200:]

x_train = train[:, :-1]
y_train = [mic2class[i] for i in train[:, -1]]

x_test = test[:, :-1]
y_test = [mic2class[i] for i in test[:, -1]]

min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []

for min_samples_leaf in min_samples_leafs:
    rf = RandomForestClassifier(min_samples_leaf=min_samples_leaf, n_jobs=-1)
    rf.fit(x_train, y_train)
    train_pred = rf.predict(x_train)
    
    accuracy = accuracy_score(y_train, train_pred)
    train_results.append(accuracy)

    y_pred = rf.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    test_results.append(accuracy)
    

line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train Accuracy')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test Accuracy')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('Accuracy score')
plt.xlabel('min_samples_leafs')
# plt.show()
plt.savefig('plots/min_samples_leafs_clf.png')
