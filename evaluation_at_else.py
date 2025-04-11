'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
MoChaTo: Evaluation script for "Pattern in neutron scattering data and the 
topology of monochain molecules"
-------------------------------------------------------------------------------
written by Jonas Soucek, 2025
at TU Dresden and Leibniz Institute of Polymer Research

This script it used to get and evaluate simulated data from an .hdf5 file. It
is part of a bachelor thesis in physics.
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


# print statement if the script is executed
print('-'*79)
print('-'*79 + '\n')
print('Executing evaluation script...')


NComps = 2                      # number of principle components to perform
                                # PCA with
TestRun = True                  # set to True if only a test run is needed:
                                # only diagramms for first condition and
                                # inter-condition evaluation will be plotted

system = 'windows'              # clearify operating systsem for file handling
if system == 'windows':
    seperator = '\\'            # define seperator for windows 
                                # operating system
elif system == 'linux':
    seperator = '/'             # define seperator for linux
                                # operating system


# root directory of the project
root_dir = '.'
# search criteria for .hdf5 files
search_crit = root_dir + '\\**\\*.hdf5'.replace('\\', seperator)


filter_obj = 'swell_sqiso_key'
eva_path = root_dir + '\\script_evaluation'.replace('\\', seperator)


for path in glob.glob(root_dir+search_crit, recursive=True):
    if not exists(eva_path):
        os.makedirs(eva_path)


    with open(eva_path + r'\results.txt', 'w') as res_file:
        res_file.write(f'File: {path}\n')
        res_file.write('This is a results file of an .hdf5 file evaluation')
        res_file.write('containing a simulation of 500 monochain molecules\n')
        res_file.write('and thier resulting structural factor regarding')
        res_file.write('neutron scattering. It contains some information on\n')
        res_file.write('the operaiting system and hardware as well as')
        res_file.write('signifficant quantities for further evaluation.\n\n')

        res_file.write(f'Number of PCA components: 2\n\n')


    DataObjs = []           # list to store all data objects
    
    
    with h5py.File(path, 'r') as file:
        file.visit(lambda x: DataObjs.append(lib.filter_func(name=x,\
                   file=file, NComps=NComps, filter_obj=filter_obj,\
                   eva_path=eva_path, system=system, TestRun=TestRun)))
        
    DataObjs = [obj for obj in DataObjs if obj is not None]     # remove None
                                                    

    # get different f for each chain length
    fn40 = [obj.f for obj in DataObjs if obj.length == 40]
    fn100 = [obj.f for obj in DataObjs if obj.length == 100]
    fn200 = [obj.f for obj in DataObjs if obj.length == 200]

    # get different root-mean-square reconstruction error for each chain length
    ren40 = [obj.mre for obj in DataObjs if obj.length == 40]
    ren100 = [obj.mre for obj in DataObjs if obj.length == 100]
    ren200 = [obj.mre for obj in DataObjs if obj.length == 200]


    # important parameters for plotting e_S
    fmin = 1.05*np.min([np.min(fn40), np.min(fn100), np.min(fn200)])\
            - 0.05*np.max([np.max(fn40), np.max(fn100), np.max(fn200)])
    fmax = 1.05*np.max([np.max(fn40), np.max(fn100), np.max(fn200)])\
            - 0.05*np.min([np.min(fn40), np.min(fn100), np.min(fn200)])
    remin = 1.05*np.min([np.min(ren40), np.min(ren100), np.min(ren200)])\
            - 0.05*np.max([np.max(ren40), np.max(ren100), np.max(ren200)])
    remax = 1.05*np.max([np.max(ren40), np.max(ren100), np.max(ren200)])\
            - 0.05*np.min([np.min(ren40), np.min(ren100), np.min(ren200)])


    # plot mean reconstruction error depending on interconnection error
    fig = plt.figure(figsize=(8, 6))                # create figure
    ax = fig.add_subplot(1, 1, 1)                   # add subplot
    # ax.axis([fmin, fmax, remin, remax])             # set axis limits

    ax.set_title(r'Relative mean reconstruction error depending on interconnection density')
    ax.set_xlabel(r'f')
    ax.set_ylabel(r'$\langle e_S\rangle$')

    ax.scatter(1/np.array(fn40), ren40,\
            s=3.5, color='dodgerblue', label=r'$n=40$')
    ax.scatter(1/np.array(fn100), ren100,\
            s=3.5, color='tomato', label=r'$n=100$')
    ax.scatter(1/np.array(fn200), ren200,\
            s=3.5, color='springgreen', label=r'$n=200$')

    ax.legend(loc='upper right')            # set legend position


    lib.save_plot(fig=fig, name='mean_recon_error',\
                  path=eva_path+seperator+'plots', system=system)


print('Evaluation finished')