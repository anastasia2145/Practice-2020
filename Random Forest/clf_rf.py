import numpy as np
import pickle

from sklearn.ensemble import RandomForestClassifier

mic2class = {0.25: 0, 1.0 : 1, 2.0: 2, 0.5: 3, 4.0: 4, 8.0: 5, 16.0: 6}

matrix = np.load("data/norm_matrix.npy")
permutation = np.load("data/permutation.npy")
matrix = matrix[permutation]
y_train = [mic2class[i] for i in matrix[:, -1]]

clf = RandomForestClassifier(
                                min_samples_split=0.1,
                                min_samples_leaf=0.1,
                                max_depth=19,
                                n_estimators=500,
                                random_state=21,
                                verbose=1,
                                #class_weight="balanced_subsample",                                
                                n_jobs=-1
                             )
clf.fit(matrix[:,:-1], y_train)

with open('models/randomforect_clf.pickle', 'wb') as f:
    pickle.dump(clf, f)
