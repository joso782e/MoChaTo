import os
import sys
sys.path.append(os.path.dirname(__file__))
from Scripts.library_MoChaTo import *
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

dataset = r'scnp_e0p15.hdf5'
filter_obj = 'swell_sqiso_key'
name = r'solvent_implicit/reactive_thermal/N_100/e_0.15/f_8/p_3/' + filter_obj

with h5py.File(dataset, 'r') as file:
    x, data = load_data(name=name, file=file, filter_obj=filter_obj)
    print(data.shape[0])