'''

'''


import os
import sys
from os.path import exists
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



def filter_func(name:str, file:h5py.File, filter_obj:str, get_datasets:list,\
                plot_path:str, system:str, TestRun:bool=False) -> None:
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
    plot_path (dtype = str)...      path to save plots
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
        # print name for evaluation in terminal
        print(f'{name}')

        # create dictionary of datasets
        data = {}

        l = len(filter_obj)                     # length of 'filter_obj' string

        # load data from object 'file' according to values in list
        # 'get_datasets'
        for dataset in get_datasets:
            data[dataset] = load_data(name=name[:-l] + dataset, file=file)


        # evaluate data of 'swell_sqiso', reduction of topological features
        sq_pca = eva_struc_fac(data=data['swell_sqiso'], princ_comps=2, transpose=False)
        
        
        # plot result of PCA on 'swell_sqiso' data (structurial factor)


        # save plot
        save_plot(fig=fig, name=name, filter_obj=filter_obj,\
                  get_dataset=dataset, plot_path=plot_path,\
                  system=system)


        # print separator line to indicate end of data group evaluation in 
        # terminal
        print('-'*79)
        

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
    m, n = data.shape[0], data.shape[1]
    data = np.reshape(data, (m, n))

    return data


def eva_struc_fac(data:np.ndarray, princ_comps:int, transpose=True)\
    -> np.ndarray:
    '''

    Function to evaluate data of structural factor

    input variables:
    data (dtype = np.ndarray)...    data group to be evaluated
    princ_comps (dtype = int)...    number of topological principle components
                                    to be reduced to
    transpos (dtype = bool)...      transpose data before evaluation

    output variables:
    principle_component (dtype = np.ndarray)...    principle components of data
    ---------------------------------------------------------------------------
    '''

    # check dtype of data
    if type(data) != np.ndarray:
        data = np.array(data)

    # transpose
    if transpose:
        data = data.transpose()


    # perform PCA on standardized data: reduce shape 'n' (number of
    # topological features) to 'princ_comps' (number of topological principle
    # components)
    pca = PCA(n_components=princ_comps)
    principle_component = pca.fit_transform(data)

    print(f'{pca.components_}')

    return principle_component

    
def save_plot(fig:plt.Figure, name:str, filter_obj:str, get_dataset:list,\
              plot_path:str, system:str) -> None:
    '''

    Function to plot data

    input variables:
    fig (dtype = plt.Figure)...     figure with ploted data
    name (dtype = str)...           name of data group
    filter_obj (dtype = str)...     name of one data group to filter simulated
                                    conditions in .hdf5 'input_file'
    get_dataset (dtype = list)...   list of data group to be loaded from
                                    'input_file'
    plot_path (dtype = str)...      path to save plots
    system (dtype = str)...         name of operating system,
                                    either 'windows' or 'linux'

    output variables:
    None
    ---------------------------------------------------------------------------
    '''   

    l = len(filter_obj)                     # length of 'filter_obj' string

    if system == 'windows':
        seperator = '\\'                     # define seperator for windows
    elif system == 'linux':
        seperator = '/'                      # define seperator for linux
    
    # define path to save figure as .png file
    path = plot_path + seperator + get_dataset
    
    # create directories if they do not exist
    if not exists(path):
        os.makedirs(path)
    
    # save figure as .png
    fig.savefig(path + seperator + name[:-l].replace('/', '&') + '.png')
    plt.pause(0.1)                                  # pause for 0.1 seconds
    plt.close()                                     # close figure