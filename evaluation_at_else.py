'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
MoChaTo: Evaluation script for "Pattern in neutron scattering data and the 
topology of monochain molecules
-------------------------------------------------------------------------------
written by Jonas Soucek, 2025
at TU Dresden and Leibniz Institute of Polymer Research

This script it used to get and evaluate simulated data from an .hdf5 file.
'''

import os
import sys
from os.path import exists
sys.path.append(os.path.dirname(__file__))
import library_MoChaTo as lib
import glob
import h5py
import numpy as np
import matplotlib.pyplot as plt


root_dir = r'C:\Users\Jonas Soucek\OneDrive\TU Dresden\Physik-Studium, Bachelor\Abschlussarbeit'
search_crit = r'\**\*.hdf5'

NComps = 2                      # number of principle components to perform
                                # PCA with

system = 'windows'              # clearify operating systsem for file handling

filter_obj = 'swell_sqiso_key'
datagroups = [filter_obj, 'swell_sqiso']
eva_path = r'C:\Users\Jonas Soucek\OneDrive\TU Dresden\Physik-Studium, Bachelor\Abschlussarbeit\script_evaluation'


for path in glob.glob(root_dir+search_crit, recursive=True):
    if not exists(eva_path):
        os.makedirs(eva_path)


    with open(eva_path + r'\results.txt', 'w') as res_file:
        res_file.write(f'File: {path}\n')
        res_file.write('This is a results file of an .hdf5 file evaluation')
        res_file.write('containing a simulation of 500 monochain molecules\n')
        res_file.write('and thier resulting structural factor regarding')
        res_file.write('neutron scattering. It contains some information on\n')
        res_file.write('the operaing system and hardware as well as')
        res_file.write('signifficant quantities for further evaluation.\n\n')

        res_file.write(f'Number of PCA components: 2\n\n')

    
    DataObjs = []

    with h5py.File(path, 'r') as file:
        file.visit(lambda x: lib.filter_func(name=x, file=file,\
            NComps=NComps, DataObjs=DataObjs, filter_obj=filter_obj,\
            eva_path=eva_path, system=system, TestRun=False))
        

print(len(DataObjs))
        

if system == 'windows':
    seperator = '\\'                    # define seperator for windows 
                                        # operating system
elif system == 'linux':
    seperator = '/'                     # define seperator for linux
                                        # operating system
                                        

# plot mean reconstruction error depending on interconnection error
'''fn40 = [obj.f for obj in DataObjs if obj.length == 40]
fn100 = [obj.f for obj in DataObjs if obj.length == 100]
fn200 = [obj.f for obj in DataObjs if obj.length == 200]

ren40 = [obj.mre for obj in DataObjs if obj.length == 40]
ren100 = [obj.mre for obj in DataObjs if obj.length == 100]
ren200 = [obj.mre for obj in DataObjs if obj.length == 200]

fig = plt.figure(figsize=(5, 3))                # create figure
ax = fig.add_subplot(1, 1, 1)                   # add subplot

ax.set_title(r'Mean reconstruction error depending on interconnection density')
ax.set_xlabel(r'1/f')
ax.set_ylabel(r'mean reconstruction error')

ax.plot(1/np.array(fn40), ren40, lw=1.0, color='blue', label=r'$n=40$')
ax.plot(1/np.array(fn100), ren100, lw=1.0, color='red', label=r'$n=100$')
ax.plot(1/np.array(fn200), ren200, lw=1.0, color='green', label=r'$n=200$')

lib.save_plot(fig=fig, name='mean_recon_error',\
              path=eva_path+seperator+'plots', system=system)'''


print('Evaluation finished')