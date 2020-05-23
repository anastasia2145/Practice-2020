import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier


matrix = np.load("data/norm_matrix.npy")
permutation = np.load("data/permutation.npy")
matrix = matrix[permutation]

y_train = [0 if i <= 4 else 1 for i in matrix[:, -1]]

clf = RandomForestClassifier(
                                min_samples_split=0.1,
                                min_samples_leaf=0.1,
                                max_depth=19,
                                n_estimators=500,
                                random_state=21,
                                verbose=1
                             )
clf.fit(matrix[:,:-1], y_train)

with open('models/randomforect_bin_clf.pickle', 'wb') as f: 
    pickle.dump(clf, f)
