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
    '\\data_evaluation\\script_evaluation\\PCA_on_qqS'.replace('\\\\', seperator)


config = {
    'NComps' : NComps,
    'binnum' : binnum,
    'root_dir' : root_dir,
    'search_crit' : search_crit,
    'filter_obj' : filter_obj,
    'eva_path' : eva_path,
    'fileformat' : '.svg',
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
from os.path import exists
import sys
sys.path.append(os.path.dirname(__file__))
import MoChaTo_unievalib as unievalib
import MoChaTo_datalib as datalib
import MoChaTo_plotlib as plotlib
import glob
import h5py
import numpy as np


# print statement if the script is executed
print('-'*79)
print('-'*79 + '\n')
print('Executing evaluation script...')


for path in glob.glob(root_dir+search_crit, recursive=True):
    if not exists(eva_path):
        os.makedirs(eva_path)


    dataObjs = []           # list to store all data objects
    
    
    with h5py.File(path, 'r') as file:
        file.visit(
            lambda x: dataObjs.append(
                datalib.filter_func(name=x, file=file)
            )
        )
        

    plotaspects = {}

    # Nrule:        - list of floats or 'all', obj.N matching rule are plotted
    # frule:        - list of floats or 'all', obj.f matching rule are plotted 
    # binnum:       - int setting number of bins 
    # title:        - str setting plot title
    # xlabel:       - str setting x-axis label
    # ylabel:       - str setting y-axis label
    # xdata:        - str referencing an obj attribute for use as xdata
    # ydata:        - str referencing an obj attribute for use as ydata
    # xlim:         - list or tuple setting x-axis view limits;
    #                 e.g. [1e-2, None]
    # ylim:         - list or tuple setting x-axis view limits,
    #                 e.g. [1e-4, None]
    # xscale:       - str setting scaling of x-axis
    # yscale:       - str setting scaling of y-axis
    # scalfac:      - list or array scaling y data befor applying Kratky
    # ls:           - list of str setting linestyles;
    #                 length must match with lw, color, marker and ms
    # lw:           - list of floats setting linewidths;
    #                 length must match with ls, color, marker and ms
    # marker:       - list of str setting marker types;
    #                 length must match with ls, lw, color and ms
    # ms:           - list of floats setting marker sizes;
    #                 length must match with ls, lw, color and marker
    # color:        - list of colors;
    #                 length must match with ls, lw, marker and ms
    # plotdomain:   - decide how to plot data as graph; only Kratky implemented
    # plot:         - decide wether to plot as graph/scatter, histogram or
    #                 errorbar
    # sortby:       - option to sort data sets by N or f values
    # legend:       - wether to display legend or not
    # legend_loc:   - str disribing the legends location in the plot
    # label:        - str for what to use in labeling data sets: either N or f
    
    plotaspects['figsize'] = (6,4)
    plotaspects['Nrule'] = [100]
    plotaspects['frule'] = [1/4]
    plotaspects['title']= r'Standard deviation of mean loop position binned over $c_2$'
    plotaspects['xlabel'] = r'$c_2 \cdot 10^3$'
    plotaspects['ylabel'] = r'$s_{n_l}$'
    plotaspects['xdata'] = 'qqSc2'
    plotaspects['ydata'] = 'loopdev'
    plotaspects['xlim'] = [None, None]
    plotaspects['ylim'] = [None, None]
    plotaspects['xscale'] = 'linear'
    plotaspects['yscale'] = 'linear'
    plotaspects['ls'] = 'None'
    plotaspects['lw'] = 1.0
    plotaspects['marker'] = 'o'
    plotaspects['ms'] = 3.5
    plotaspects['color'] = 'dodgerblue'
    plotaspects['plotdomain'] = 'PCspace'
    plotaspects['plot'] = 'errorbar'
    plotaspects['sortby'] = 'f'
    plotaspects['legend'] = False
    plotaspects['legend_loc'] = 'upper left'
    plotaspects['label'] = False
    plotaspects['binnum'] = 50


    # remove None and select relevant data objects
    if type(plotaspects['Nrule']) is list and type(plotaspects['frule']) is list:
        dataObjs = [
            obj for obj in dataObjs if obj and (
                obj.N in plotaspects['Nrule'] and obj.f in plotaspects['frule']
            )
        ]
    elif type(plotaspects['Nrule']) is list:
        dataObjs = [
            obj for obj in dataObjs if obj and obj.N in plotaspects['Nrule']
        ]
    elif type(plotaspects['frule']) is list:
        dataObjs = [
            obj for obj in dataObjs if obj and obj.f in plotaspects['frule']
        ]
    else:
        dataObjs = [obj for obj in dataObjs if obj]


    for obj in dataObjs:
        unievalib.ManipulateData(
            obj, ['q', 'q', 'S'], 'kratkyS', operant='*'
        )
        unievalib.PerfPCA(
            obj, 'kratkyS', n_components=NComps
        )


    evaplot = plotlib.PlotData(plotaspects, dataObjs)
    evaplot.CreateFigure(
        title=plotaspects['title'], xLabel=plotaspects['xlabel'],
        yLabel=plotaspects['ylabel'], figsize=plotaspects['figsize']
    )


print('Evaluation finished')