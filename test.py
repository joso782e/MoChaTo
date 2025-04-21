import os
import sys
sys.path.append(os.path.dirname(__file__))
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

n = np.diag([1, 2, 3])

m = np.array([1, 2, 3])

print(n-m)