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
eva_path = root_dir + '\\data_evaluation\\script_evaluation'.replace('\\\\', seperator)


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

plotaspects = {
    'Nrule' : [40],
    'frule' : [1/2],
    'binnum' : 30,
    'title' : 'Plot title',
    'xlabel' : 'xlable',
    'ylabel' : 'ylable',
    'xdata' : 'q',
    'ydata' : 'S',
    'xlim' : [1e-2, None],
    'ylim' : [1e-4, None],
    'xscale' : 'linear',
    'yscale' : 'linear',
    'scalfac' : 1.0,
    'lw' : [1.0],
    'ls' : ['-'],
    'color' : ['dodgerblue'],
    'marker' :['o'],
    'markersize' : [3.5],
    'plotdomain' : 'Kratky',
    'plot' : 'diag',
    'sortby' : 'N',
    'legend' : True,
    'legend_loc' : 'upper right',
    'label' : 'legend label'
}


def FitGyraRad(x, a0, a1):
    '''
    Function for fitting radii of gyration to form factor 
    '''
    return a0 - 1/3*(a1*x)**2


import json

with open(
    '.\\data_evaluation\\Scripts\\config.json'.replace('\\\\', seperator),
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

    for obj in DataObjs:
        # perform fit for radii of gyration
        obj.PerfFit(FitGyraRad, 'q', 'S', 'Rg')
    
    plotaspects['title'] = 'Radii of gyration vs. PC1'
    plotaspects['xlabel'] = r'$c_1$'
    plotaspects['ylabel'] = r'$R_g$'
    plotaspects['xdata'] = 'c1'
    plotaspects['ydata'] = 'Rg1'
    plotaspects['xlim'] = [None, None]
    plotaspects['ylim'] = [None, None]
    plotaspects['ls'] = ['None']
    plotaspects['plotdomain'] = 'PCspace'
    plotaspects['legend'] = False

    gyraplot = lib.PlotData(plotaspects, DataObjs)
    gyraplot.CreatePlot()

print('Evaluation finished')