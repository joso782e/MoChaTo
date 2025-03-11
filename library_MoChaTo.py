'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
MoChaTo:
-------------------------------------------------------------------------------
written by Jonas Soucek, 2025
at TU Dresden and Leibniz Institute of Polymer Research
'''


import os
import sys
from os.path import exists
import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def filter_func(name:str, file:h5py.File, filter_obj:str, get_datasets:list,\
                eva_path:str, system:str, TestRun:bool=False) -> None:
    '''

    Function to filter data groups in .hdf5 'input_file' that contain the
    string 'filter_obj' and perform data collection, evaluation and plotting
    on them

    input variables:
    name (dtype = str)...           name of data group to be filtered
    file (dtype = h5py.File)...     input file
    filter_obj (dtype = str)...     name of one data group to filter simulated
                                    conditions in .hdf5 'input_file'
    get_datasets (dtype = list)...  list of data groups to be loaded from .hdf5
                                    'input_file'
    eva_path (dtype = str)...       path to save evaluation results and plots
    system (dtype = str)...         name of operating system,
                                    either 'windows' or 'linux'
    TestRun (dtype = bool)...       flag to exit script after evaluation of
                                    first data group

    output variables:
    None
    ---------------------------------------------------------------------------
    '''

    # perform data collection, evaluation and plotting for each simulated
    # condition
    if filter_obj in name:
        # print name in terminal for debugging
        print(f'{name}')

        # create dictionary of datasets for current condition
        data = {}

        l = len(filter_obj)                     # length of 'filter_obj' string

        # load data from object 'file' according to values in list
        # 'get_datasets'
        for dataset in get_datasets:
            data[dataset] = load_data(name=name[:-l] + dataset, file=file)


        # perform PCA on 'swell_sqiso' data
        pca = PCA(n_components=2)                       # define PCA object
        pca.fit(data['swell_sqiso'])                    # fit PCA object to 
                                                        # data
        components = pca.components_                    # get principle 
                                                        # components
        transf = pca.transform(data['swell_sqiso'])     # transform data to
                                                        # principle components

        
        # calculate reconstraction error of PCA
        recon = np.matmul(transf, components)\
                + np.mean(data['swell_sqiso'], axis=0)
        recon_er = np.sqrt(np.mean((data['swell_sqiso'] - recon)**2))


        # important parameters for plotting
        qmin = 1.05*np.min(data['swell_sqiso_key'])\
               - 0.05*np.max(data['swell_sqiso_key'])
        qmax = 1.05*np.max(data['swell_sqiso_key'])\
               - 0.05*np.min(data['swell_sqiso_key'])
        Smin = 1.05*np.min([components[0,:]*data['swell_sqiso_key']**2,\
                            components[1,:]*data['swell_sqiso_key']**2])\
               - 0.05*np.max([components[0,:]*data['swell_sqiso_key']**2,\
                              components[1,:]*data['swell_sqiso_key']**2])
        Smax = 1.05*np.max([components[0,:]*data['swell_sqiso_key']**2,\
                            components[1,:]*data['swell_sqiso_key']**2])\
               - 0.05*np.min([components[0,:]*data['swell_sqiso_key']**2,\
                              components[1,:]*data['swell_sqiso_key']**2])
        
        # plot result of PCA on 'swell_sqiso' data (structurial factor),
        # component 1 and 2 dependend on 'swell_sqiso_key'
        fig = plt.figure(figsize=(8, 6))            # create figure
        ax = fig.add_subplot(1, 1, 1)               # add subplot
        ax.axis([qmin, qmax, Smin, Smax])           # set axis limits

        ax.set_title(r'Principle components dependend on $q$')
        ax.set_xlabel(r'$q$')
        ax.set_ylabel(r'$S_1(q)q^2$')

        ax.plot(data['swell_sqiso_key'],\
                  components[0,:]*data['swell_sqiso_key']**2, lw=1.0,\
                  color='blue', label='Component 1')
        ax.plot(data['swell_sqiso_key'],\
                  components[1,:]*data['swell_sqiso_key']**2, lw=1.0,\
                  color='red', label='Component 2')
        
        ax.legend(loc='upper right')

        # save and close figure
        save_plot(fig=fig, name=name, filter_obj=filter_obj,\
                  eva_path=eva_path, eva_aspect='PCA_comp_plot',\
                  system=system)
        

        # plot result of PCA on 'swell_sqiso' data (structurial factor),
        # component 1 vs. component 2 of transformed data
        fig = plt.figure(figsize=(5, 3))                # create figure
        ax = fig.add_subplot(1, 1, 1)                   # add subplot

        ax.set_title(r'2D scatter plot of PCA-transformed data')
        ax.set_xlabel(r'Component 1')
        ax.set_ylabel(r'Component 2')

        ax.scatter(transf[:,0], transf[:,1], s=0.5, c='blue')

        # save and close figure
        save_plot(fig=fig, name=name, filter_obj=filter_obj,\
                  eva_path=eva_path, eva_aspect='comp1_comp2_transf_data',\
                  system=system)

        
        if system == 'windows':
            seperator = '\\'                    # define seperator for windows
                                                # operating system
        elif system == 'linux':
            seperator = '/'                     # define seperator for linux
                                                # operating system

        # print and save results in seperate .txt file
        with open(eva_path + seperator + 'results.txt', 'a') as res_file:
            res_file.write(f'{name[:-l]}\n')
            res_file.write('Mean reconstruction error:\n')
            res_file.write(f'{round(recon_er,6)}\n')
            res_file.write(f'By component 1 explained variance:\n')
            res_file.write(f'{round(pca.explained_variance_ratio_[0],6)}\n')
            res_file.write(f'By component 2 explained variance:\n')
            res_file.write(f'{round(pca.explained_variance_ratio_[1],6)}\n')
            res_file.write('-'*79 + '\n\n')

        # print separator line to indicate end of one condition evaluation in 
        # terminal during debugging
        # print('-'*79)
        

        if TestRun:
            sys.exit()          # exit script if TestRun flag is set to True
        
        
def load_data(name:str, file:h5py.File) -> np.ndarray:
    '''

    Function to load data groups from .hdf5 'input_file'

    input variables:
    name (dtype = str)...           name of data group to be filtered
    file (dtype = h5py.File)...     input file
    filter_obj (dtype = str)...     name of one data group to filter simulated
                                    conditions in .hdf5 'input_file'
    get_dataset (dtype = str)...    name of data group to be loaded from
                                    'input_file'

    output variables:
    data (dtype = np.ndarray)...    data loaded from 'input_file'
    ---------------------------------------------------------------------------
    '''

    # load data from dataset 'swell_sqiso'
    data = file[name]

    # get shape of data and reshape it from (m, n, 1) to (m, n)
    data = np.squeeze(data)

    return data

    
def save_plot(fig:plt.Figure, name:str, filter_obj:str, eva_path:str,\
              eva_aspect:str, system:str) -> None:
    '''

    Function to plot data

    input variables:
    fig (dtype = plt.Figure)...     figure with ploted data
    name (dtype = str)...           name of data group
    filter_obj (dtype = str)...     name of one data group to filter simulated
                                    conditions in .hdf5 'input_file'
    eva_path (dtype = str)...       path to save evaluation results and plots
    eva_aspect (dtype = str)...     name of evaluation aspect to be ploted
    system (dtype = str)...         name of operating system,
                                    either 'windows' or 'linux'

    output variables:
    None
    ---------------------------------------------------------------------------
    '''   

    l = len(filter_obj)                     # length of 'filter_obj' string

    if system == 'windows':
        seperator = '\\'                    # define seperator for windows 
                                            # operating system
    elif system == 'linux':
        seperator = '/'                     # define seperator for linux
                                            # operating system
    
    # define path to save figure as .png file
    path = eva_path + seperator +'plots' + seperator + eva_aspect
    
    # create directories if they do not exist
    if not exists(path):
        os.makedirs(path)
    
    # save figure as .png
    fig.savefig(path + seperator + name[:-l].replace('/', '&') + '.png')
    plt.pause(0.1)                                  # pause for 0.1 seconds
    plt.close()                                     # close figure