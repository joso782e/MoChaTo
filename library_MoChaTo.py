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


class FileData:
    '''
    Class to store and update results for comprehensive file evaluation after 
    individual file evaluation.
    '''
    # define contructor
    def __init__(self, Cond:str, NChain:int, InterconDens:int,\
                 qKey:np.ndarray, SData:np.ndarray, NComps:int):
        self.condition = Cond           # string describing simulated condition
        self.length = NChain            # chainlength in number of spheres
        self.f = InterconDens           # connector distance
        self.q = qKey                   # values in q-space
        self.S = SData                  # structural factor in q-space
        self.ncomps = NComps            # number of principle components
        self.reconS = None              # reconstructed data from PCA
        self.comps = None               # principle components, sorted via
                                        # explained variance
        self.var = None                 # explained variance, sorted from
                                        # highets to lowest
        self.varratio = None            # percentage of explained variance
        self.mre = None                 # mean reconstraction error
        self.re = None                  # mean reconstruction error in q-space

    @staticmethod
    def ExtractFileData(file:h5py.File, path:str, filter_obj:str, ncomps:int):

        l = len(filter_obj)         # length of filter_object
        Cond = path[:-l].replace('/', '&')

        if 'N_40' in path:
            NChain = 40
        elif 'N_100' in path:
            NChain = 100
        elif 'N_200' in path:
            NChain = 200

        if 'f_2' in path:
            InterconDens = 2
        elif 'f_3' in path:
            InterconDens = 3
        elif 'f_4' in path:
            InterconDens = 4
        elif 'f_5' in path:
            InterconDens = 5
        elif 'f_6' in path:
            InterconDens = 6
        elif 'f_7' in path:
            InterconDens = 7
        elif 'f_8' in path:
            InterconDens = 8
        elif 'f_12' in path:
            InterconDens = 12
        elif 'f_18' in path:
            InterconDens = 18
        elif 'f_23' in path:
            InterconDens = 23

        # get 'swell_sqiso_key' group and reshape it from (m, 1) to (m)
        qKey = np.squeeze(file[path])

        # get 'swell_sqiso' group and reshape it from (m, n, 1) to (m, n)
        SData = np.squeeze(file[path[:-l]+'swell_sqiso'])

        return FileData(Cond=Cond, NChain=NChain, InterconDens=, qKey=qKey, SData=SData, NComps=ncomps)


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

        DataObj = FileData.ExtractFileData(file=file, path=name,\
                                           filter_obj=filter_obj,\
                                           ncomps=NComps)


        # perform PCA on 'swell_sqiso' data
        pca = PCA(n_components=DataObj.ncomps)          # set PCA object
        pca.fit(FileObj.S)                              # fit PCA object to 
                                                        # data
        DataObj.comps = pca.components_                 # get principle 
                                                        # components
        transf = pca.transform(DataObj.S)               # transform data to
                                                        # principle components

        
        # calculate reconstraction error of PCA via root-mean-square method
        DataObj.reconS = np.matmul(transf, DataObj.comps)\
                         + np.mean(DataObj.S, axis=0)
        DataObj.mre = np.sqrt(np.mean((DataObj.S - DataObj.reconS)**2))
        DataObj.re = np.sqrt(np.mean((DataObj.S - DataObj.reconS)**2, axis=0))


        # plot principle components in q-space
        plot_princ_comps(data=data, components=components, name=name,\
                         filter_obj=filter_obj, eva_path=eva_path,\
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

    # define path to safe plot depending on chain length
    if 'N_40' in name:
        path = eva_path + seperator +'plots'  + seperator + eva_aspect\
               + seperator + 'N_40'
    elif 'N_100' in name:
        path = eva_path + seperator +'plots' + seperator + eva_aspect\
               + seperator + 'N_100'
    elif 'N_200' in name:
        path = eva_path + seperator +'plots' + seperator + eva_aspect\
               + seperator + 'N_200'

    
    # create directories if they do not exist
    if not exists(path):
        os.makedirs(path)
    
    # save figure as .png
    fig.savefig(path + seperator + name + '.png')
    plt.pause(0.1)                                  # pause for 0.1 seconds
    plt.close()                                     # close figure


def plot_princ_comps(data:dict, components:np.ndarray, name:str,\
                     filter_obj:str, eva_path:str, system:str)\
    -> None:
    '''
    Function to make script more clear. It contains all lines regarding
    the plot of the principle components.
    '''

    # important parameters for plotting principle components
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