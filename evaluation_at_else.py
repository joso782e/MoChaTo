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
NComps = 2                      # number of principle components to perform
                                # PCA with
binnum = 30                     # number of bins for hinstogramm plots
TestRun = True                  # set to True if only a test run is needed:
                                # only diagramms for first condition and
                                # inter-condition evaluation will be plotted

system = 'windows'              # clearify operating systsem for file handling
if system == 'windows':
    seperator = '\\\\'            # define seperator for windows 
                                # operating system
elif system == 'linux':
    seperator = '/'             # define seperator for linux
                                # operating system


# root directory of the project
root_dir = '.'
# search criteria for .hdf5 files
search_crit = root_dir + '\\**\\*.hdf5'.replace('\\\\', seperator)


filter_obj = 'swell_sqiso_key'
eva_path = root_dir + '.\\data_evaluation\\script_evaluation'.replace('\\\\', seperator)


config = {
    'NComps' : NComps,
    'binnum' : binnum,
    'root_dir' : root_dir,
    'search_crit' : search_crit,
    'filter_obj' : filter_obj,
    'eva_path' : eva_path,
    'fileformat' : '.png',
    'system' : system,
}

Nrule = 40
frule = 23
binnum = 30
title = 'Plot title'
xlabel = 'xlable'
ylabel = 'ylable'
xdata = 'q'
ydata = 'S'
xlim = [1e-2, None]
ylim = [1e-4, None]
xscale = 'linear'
yscale = 'linear'
lw = 1.0
ls = '-'
color = 'dodgerblue'
marker='o'
ms = 3.5
plotdomain = 'Kratky'
plot = 'diag'
seperate_plots = True
sortby = 'N'
legend = True
legend_loc = 'upper right'
label = 'legend label'


def FitGyraRad(x, a, b):
    '''
    Function for fitting radii of gyration to form factor 
    '''
    return a - b*x**2


import json

with open(
    '.\\data_evaluation\\Scripts\config.json'.replace('\\\\', seperator),
    'w') as configf:
    json.dump(config, configf)


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
                   file=file)))
        
    DataObjs = [obj for obj in DataObjs if obj is not None]     # remove None

    for DataObj in DataObjs:
        if not TestRun:
            # plot principle components in q-space
            lib.plot_princ_comps(DataObj=DataObj)

            # plot structural factor in PC-space
            lib.plot_data_PCspace(DataObj=DataObj)

            # plot reconstruction error in q-space
            lib.plot_recon_error(DataObj=DataObj)

            # plot example curves, reconstructed curves and mean curve in
            # q-space
            lib.plot_data_qspace(DataObj=DataObj)
            
            # plot histogramm of loop length, problem: data to large
            lib.plot_ll_hist(DataObj=DataObj)

            # plot histogramm of mean loop length
            lib.plot_mll_hist(DataObj=DataObj)

            # plot mean loop length against c1
            lib.plot_mll_vs_ci(DataObj=DataObj, i=1)

            # plot mean loop length against c2
            lib.plot_mll_vs_ci(DataObj=DataObj, i=2)


    # get different f for each chain length
    fn40 = [obj.f for obj in DataObjs if obj.length == 40]
    fn100 = [obj.f for obj in DataObjs if obj.length == 100]
    fn200 = [obj.f for obj in DataObjs if obj.length == 200]

    # get different root-mean-square reconstruction error for each chain length
    ren40 = [obj.mre for obj in DataObjs if obj.length == 40]
    ren100 = [obj.mre for obj in DataObjs if obj.length == 100]
    ren200 = [obj.mre for obj in DataObjs if obj.length == 200]

    # varianze of reconstruction error for each chain length
    var40 = [np.sqrt(obj.mrevar) for obj in DataObjs if obj.length == 40]
    var100 = [np.sqrt(obj.mrevar) for obj in DataObjs if obj.length == 100]
    var200 = [np.sqrt(obj.mrevar) for obj in DataObjs if obj.length == 200]

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

    ax.errorbar(1/np.array(fn40), ren40, yerr=var40, errorevery=(0, 3),\
                ls='None', marker='o', ms=3.5, mfc='dodgerblue',\
                mec='dodgerblue', label=r'$n=40$')
    ax.errorbar(1/np.array(fn100), ren100, yerr=var100, errorevery=(0, 3),\
                ls='None', marker='o', ms=3.5, mfc='tomato', mec='tomato',\
                label=r'$n=100$')
    ax.errorbar(1/np.array(fn200), ren200, yerr=var200, errorevery=(1, 3),\
                ls='None', marker='o', ms=3.5, mfc='springgreen',\
                mec='springgreen', label=r'$n=200$')

    ax.legend(loc='upper right')            # set legend position


    lib.save_plot(fig=fig, name='mean_recon_error',\
                  path=eva_path+seperator+'plots', system=system)


print('Evaluation finished')