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
eva_path = root_dir +\
    '\\data_evaluation\\script_evaluation'.replace('\\\\', seperator)


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

        # perform PCA on form factor
        obj.PerfPCA(setname='S')
        obj.PerfRecon(setname='S')

    
    plotaspects = {}

    # Nrule:        -list of floats or 'all', obj.N matching rule are plotted
    # frule:        -list of floats or 'all', obj.f matching rule are plotted 
    # binnum:       -int setting number of bins 
    # title:        -str setting plot title
    # xlabel:       -str setting x-axis label
    # ylabel:       -str setting y-axis label
    # xdata:        -str referencing an obj attribute for use as xdata
    # ydata:        -str referencing an obj attribute for use as ydata
    # xlim:         -list or tuple setting x-axis view limits;
    #                e.g. [1e-2, None]
    # ylim:         -list or tuple setting x-axis view limits,
    #                e.g. [1e-4, None]
    # xscale:       -str setting scaling of x-axis
    # yscale:       -str setting scaling of y-axis
    # scalfac:      -list or array scaling y data
    # ls:           -list of str setting linestyles;
    #                length must match with lw, color, marker and ms
    # lw:           -list of floats setting linewidths;
    #                length must match with ls, color, marker and ms
    # marker:       -list of str setting marker types;
    #                length must match with ls, lw, color and ms
    # ms:           -list of floats setting marker sizes;
    #                length must match with ls, lw, color and marker
    # color:        -list of colors;
    #                length must match with ls, lw, marker and ms
    # plotdomain:   -decide how to plot data as graph; only Kratky implemented
    # plot:         -decide wether to plot as graph/scatter or as histogram
    # sortby:       -option to sort data sets by N or f values
    # legend:       -wether to display legend or not
    # legend_loc:   -str disribing the legends location in the plot
    # label:        -str for what to use in labeling data sets: either N or f
    
    plotaspects['Nrule'] = [40, 100, 200]
    plotaspects['frule'] = [1/18]
    plotaspects['title'] = 'Radii of gyration vs. PC1'
    plotaspects['xlabel'] = r'$c_1$'
    plotaspects['ylabel'] = r'$R_g$'
    plotaspects['xdata'] = 'c1'
    plotaspects['ydata'] = 'Rg1'
    plotaspects['xlim'] = [None, None]
    plotaspects['ylim'] = [None, None]
    plotaspects['xscale'] = 'linear'
    plotaspects['yscale'] = 'linear'
    plotaspects['scalfac'] = 1.0
    plotaspects['ls'] = ['None' for i in plotaspects['Nrule']]
    plotaspects['lw'] = [1.0 for i in plotaspects['Nrule']]
    plotaspects['marker'] = ['o' for i in plotaspects['Nrule']]
    plotaspects['ms'] = [3.0 for i in plotaspects['Nrule']]
    plotaspects['color'] = ['dodgerblue', 'limegreen', 'orangered']
    plotaspects['plotdomain'] = 'PCspace'
    plotaspects['sortby'] = 'N'
    plotaspects['legend'] = True
    plotaspects['legend_loc'] = 'upper left'
    plotaspects['label'] = 'N'

    gyraplot = lib.PlotData(plotaspects, DataObjs)
    gyraplot.CreatePlot()

print('Evaluation finished')