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
        self.condi = Cond               # string describing simulated condition
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

        return FileData(Cond=Cond, NChain=NChain, InterconDens=InterconDens,\
                        qKey=qKey, SData=SData, NComps=ncomps)


def filter_func(name:str, file:h5py.File, NComps:int, DataObjs:list,\
                filter_obj:str, eva_path:str, system:str, TestRun:bool=False)\
    -> None:
    '''
    Function to filter data groups in .hdf5 'input_file' that contain the
    string 'filter_obj' and perform data collection, evaluation and plotting
    on them

    input variables:
    name (dtype = str)...           name of data group to be filtered
    file (dtype = h5py.File)...     input file
    NComps (dtype = int)...         number of principal components to perform 
                                    PCA with
    filter_obj (dtype = str)...     name of one data group to filter simulated
                                    conditions in .hdf5 'input_file'
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
        pca = PCA(n_components=DataObj.ncomps)              # set PCA object
        pca.fit(DataObj.S)                                  # fit PCA object
                                                            # to data
        DataObj.comps = pca.components_                     # get principle 
                                                            # components
        DataObj.varratio = pca.explained_variance_ratio_    # get explained
                                                            # variance ratio
        transf = pca.transform(DataObj.S)                   # transform data to

        
        # calculate reconstraction error of PCA via root-mean-square method
        DataObj.reconS = np.matmul(transf, DataObj.comps)\
                         + np.mean(DataObj.S, axis=0)
        DataObj.mre = np.sqrt(np.mean((DataObj.S - DataObj.reconS)**2))
        DataObj.re = np.sqrt(np.mean((DataObj.S - DataObj.reconS)**2, axis=0))


        # plot principle components in q-space
        plot_princ_comps(DataObj=DataObj, eva_path=eva_path,  system=system)
        

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
            res_file.write(f'{DataObj.condi}\n')
            res_file.write('Mean reconstruction error:\n')
            res_file.write(f'{round(DataObj.mre,6)}\n')
            res_file.write(f'Variance ratio explained by component 1:\n')
            res_file.write(f'{round(DataObj.varratio[0],6)}\n')
            res_file.write(f'Variance ratio explained by component 2:\n')
            res_file.write(f'{round(DataObj.varratio[1],6)}\n')
            res_file.write('-'*79 + '\n\n')

        # print separator line to indicate end of one condition evaluation in 
        # terminal during debugging
        # print('-'*79)
        
        DataObjs.append(DataObj)

        if TestRun:
            sys.exit()          # exit script if TestRun flag is set to True

    
def save_plot(fig:plt.Figure, name:str, path:str, system:str,\
              fileformat:str='.png') -> None:
    '''
    Function to safe plot

    input variables:
    fig (dtype = plt.Figure)...     figure with ploted data
    name (dtype = str)...           name of plot file
    path (dtype = str)...           path to safe plot file
    system (dtype = str)...         name of operating system,
                                    either 'windows' or 'linux'
    fileformat (dtype = str)...     file format to safe plot file to

    output variables:
    None
    ---------------------------------------------------------------------------
    '''   
    if system == 'windows':
        seperator = '\\'                    # define seperator for windows 
                                            # operating system
    elif system == 'linux':
        seperator = '/'                     # define seperator for linux
                                            # operating system
    
    # create directories if they do not exist
    if not exists(path):
        os.makedirs(path)
    
    # save figure as .png
    fig.savefig(path + seperator + name + fileformat)
    plt.pause(0.1)                                  # pause for 0.1 seconds
    plt.close()                                     # close figure


def plot_princ_comps(DataObj:FileData, eva_path:str, system:str)\
    -> None:
    '''
    Function to make script more clear. It contains all lines regarding
    the plot of the principle components.
    '''

    # important parameters for plotting principle components
    qmin = 1.05*np.min(DataObj.q) - 0.05*np.max(DataObj.q)
    qmax = 1.05*np.max(DataObj.q) - 0.05*np.min(DataObj.q)
    Smin = 1.05*np.min([DataObj.comps[0,:]*DataObj.q**2,\
                        DataObj.comps[1,:]*DataObj.q**2])\
            - 0.05*np.max([DataObj.comps[0,:]*DataObj.q**2,\
                            DataObj.comps[1,:]*DataObj.q**2])
    Smax = 1.05*np.max([DataObj.comps[0,:]*DataObj.q**2,\
                        DataObj.comps[1,:]*DataObj.q**2])\
            - 0.05*np.min([DataObj.comps[0,:]*DataObj.q**2,\
                            DataObj.comps[1,:]*DataObj.q**2])
    
    # plot result of PCA on 'swell_sqiso' data (structurial factor),
    # component 1 and 2 dependend on 'swell_sqiso_key'
    fig = plt.figure(figsize=(8, 6))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot
    ax.axis([qmin, qmax, Smin, Smax])           # set axis limits

    ax.set_title(r'Principle components dependend on $q$')
    ax.set_xlabel(r'$q$')
    ax.set_ylabel(r'$S_1(q)q^2$')

    ax.plot(DataObj.q, DataObj.comps[0,:]*DataObj.q**2, lw=1.0,\
            color='blue', label='Component 1')
    ax.plot(DataObj.q, DataObj.comps[1,:]*DataObj.q**2, lw=1.0,\
            color='red', label='Component 2')
    
    ax.legend(loc='upper right')


    if system == 'windows':
        seperator = '\\'                    # define seperator for windows 
                                            # operating system
    elif system == 'linux':
        seperator = '/'                     # define seperator for linux
                                            # operating system

    # define path to safe plot depending on chain length
    if DataObj.f == 40:
        path = eva_path + seperator +'plots'  + seperator + 'PCA_comp_plot'\
               + seperator + 'N_40'
    elif DataObj.f == 100:
        path = eva_path + seperator +'plots' + seperator + 'PCA_comp_plot'\
               + seperator + 'N_100'
    elif DataObj.f == 200:
        path = eva_path + seperator +'plots' + seperator + 'PCA_comp_plot'\
               + seperator + 'N_200'

    # save and close figure
    save_plot(fig=fig, name=DataObj.condi, path=path, system=system)