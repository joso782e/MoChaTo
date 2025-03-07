import os
import sys
sys.path.append(os.path.dirname(__file__))
from library_MoChaTo import *
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

datafile = r'C:\Users\jonas\OneDrive\TU Dresden\Physik-Studium, Bachelor\Abschlussarbeit\scnp_e0p15.hdf5'
filter_obj = 'swell_sqiso_key'
datagroups = [filter_obj, 'swell_sqiso']
name = r'solvent_implicit/reactive_thermal/N_100/e_0.15/f_8/p_3/'

data = {}

with h5py.File(datafile, 'r') as file:
    for dataset in datagroups:
        data[dataset] = load_data(name=name+dataset, file=file)
    
    pca = PCA(n_components=2)
    pca.fit(data['swell_sqiso'])
    print(f'{pca.components_}')    