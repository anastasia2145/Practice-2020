import numpy as np
import pickle

from sklearn.ensemble import RandomForestRegressor

mic2class = {0.25: 0, 1.0 : 1, 2.0: 2, 0.5: 3, 4.0: 4, 8.0: 5, 16.0: 6}

matrix = np.load("data/norm_matrix.npy")
permutation = np.load("data/permutation.npy")
matrix = matrix[permutation]

reg = RandomForestRegressor(
                                min_samples_split=0.01,
                                min_samples_leaf=0.01,
                                max_depth=6,
                                n_estimators=100,
                                random_state=21,
                                verbose=1,
                                n_jobs=-1
                            )

reg.fit(matrix[:,:-1], matrix[:,-1])

with open('models/randomforect_reg_100n_6md_ss01_sl01.pickle', 'wb') as f:
    pickle.dump(reg, f)
