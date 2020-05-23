import os
import re
import time
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from random import sample
from collections import Counter

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA

kmers = open("columns.txt").read().split()
files = open("rows.txt").read().split()
matrix = np.load("norm_matrix.npy")
#matrix = matrix[:100]
print("Matrix load")
permutation = sample(range(0, len(files)), len(files))

files = np.array(files)
files = files[permutation]
matrix = matrix[permutation]
print("Permutation applied")
x, y = matrix[:, :-1], matrix[:, -1]

print("Before train")
model = CatBoostRegressor(iterations=10,
                          learning_rate=0.1,
                          depth=6,
                          loss_function='RMSE',
                          random_seed=None,
                          verbose=True
                         )
try:                         
    fit_model = model.fit(x, y)                        
except Exception as e:
    print(e)
# bootstrap_type="Bayesian"
# rsm=0.01,
print("Saving model...")
fit_model.save_model("fit_model")

