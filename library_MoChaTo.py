'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
MoChaTo:
-------------------------------------------------------------------------------
written by Jonas Soucek, 2025
at TU Dresden and Leibniz Institute of Polymer Research
'''


import os
from os.path import exists
import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# print import statement if the module has been imported successfully
print("Module \"library_MoChaTo.py\" imported successfully")


class FileData:
    '''
    Class to store and update results for comprehensive file evaluation after 
    individual file evaluation.
    '''
    # define contructor
    def __init__(self, Cond:str, NChain:int, InterconDens:int,\
                 qKey:np.ndarray, SData:np.ndarray, crosslinks:np.ndarray,\
                 NComps:int):
        self.condi = Cond                   # string describing simulated
                                            # condition
        self.length = NChain                # chainlength in number of spheres
        self.f = InterconDens               # connector distance
        self.q = qKey                       # values in q-space
        self.S = SData                      # structural factor in q-space
        self.clmat = crosslinks             # crosslink matrix
        self.mS = np.mean(SData, axis=0)    # mean of structural factor in
                                            # q-space
        self.ncomps = NComps                # number of principle components
        self.PCspaceS = None                # structural factor in PC-space
        self.reconS = None                  # reconstructed data from PCA
        self.comps = None                   # principle components, sorted via
                                            # explained variance
        self.var = None                     # explained variance, sorted from
                                            # highets to lowest
        self.varratio = None                # percentage of explained variance
        self.mre = None                     # relative mean reconstraction
                                            # error
        self.re = None                      # relative reconstruction error in 
                                            # q-space


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

        if 'f_23' in path:
            InterconDens = 23
        elif 'f_18' in path:
            InterconDens = 18
        elif 'f_12' in path:
            InterconDens = 12
        elif 'f_8' in path:
            InterconDens = 8
        elif 'f_7' in path:
            InterconDens = 7
        elif 'f_6' in path:
            InterconDens = 6
        elif 'f_5' in path:
            InterconDens = 5
        elif 'f_4' in path:
            InterconDens = 4
        elif 'f_3' in path:
            InterconDens = 3
        elif 'f_2' in path:
            InterconDens = 2
        

        # get 'swell_sqiso_key' group and reshape it from (m, 1) to (m) and
        # conversion to experimental data
        qKey = np.squeeze(file[path])/5.19

        # get 'swell_sqiso' group and reshape it from (m, n, 1) to (m, n)
        SData = np.squeeze(file[path[:-l]+'swell_sqiso'])

        # get crosslink matrix
        crosslinks = file[path[:-l]+'crosslinks']


        return FileData(Cond=Cond, NChain=NChain, InterconDens=InterconDens,\
                        qKey=qKey, SData=SData, crosslinks=crosslinks,\
                        NComps=ncomps)
    

    def mll(self) -> np.ndarray:
        '''
        Function to calculate the mean loop length for each molecule
        '''

        mll = np.abs(self.clmat[1] - self.clmat[2])

        return np.mean(mll, axis=1)


def filter_func(name:str, file:h5py.File, NComps:int, filter_obj:str,\
                eva_path:str, system:str, TestRun:bool=False)\
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
    TestRun (dtype = bool)...       flag wether to plot data for condition
                                    evaluation or not

    output variables:
    DataObj or None
    ---------------------------------------------------------------------------
    '''
    # perform data collection, evaluation and plotting for each simulated
    # condition
    if filter_obj in name:
        # print name in terminal for debugging
        print(f'{name}')


        # define seperator for file handling depending on operating system
        if system == 'windows':
            seperator = '\\'                    # define seperator for windows
                                                # operating system
        elif system == 'linux':
            seperator = '/'                     # define seperator for linux
                                                # operating system


        DataObj = FileData.ExtractFileData(file=file, path=name,\
                                           filter_obj=filter_obj,\
                                           ncomps=NComps)


        # perform PCA on 'swell_sqiso' data
        # create PCA object
        pca = PCA(n_components=DataObj.ncomps)
        # fit PCA object to data
        pca.fit(DataObj.S)
        # get principle components
        DataObj.comps = pca.components_
        # get explained variance ratio
        DataObj.varratio = pca.explained_variance_ratio_
        # transform data to principel component space
        DataObj.PCspaceS = pca.transform(DataObj.S)

        
        # calculate reconstraction error of PCA via root-mean-square method
        # can i calculate that in the class? (ask Marco)
        DataObj.reconS = np.matmul(DataObj.PCspaceS, DataObj.comps)\
                         + DataObj.mS
        DataObj.mre = np.sqrt(np.mean((DataObj.S - DataObj.reconS)**2))\
                      /np.mean(DataObj.mS)
        DataObj.re = np.sqrt(np.mean((DataObj.S - DataObj.reconS)**2, axis=0))\
                     /DataObj.mS
        

        if not TestRun:
            # plot principle components in q-space
            plot_princ_comps(DataObj=DataObj, eva_path=eva_path,  system=system)


            # plot structural factor in PC-space
            plot_data_PCspace(DataObj=DataObj, eva_path=eva_path,\
                              system=system)


            # plot reconstruction error in q-space
            plot_recon_error(DataObj=DataObj, eva_path=eva_path, system=system)


            # plot example curves, reconstructed curves and mean curve in
            # q-space
            plot_data_qspace(DataObj=DataObj, eva_path=eva_path,\
                                system=system)


        # print and save results in seperate .txt file
        with open(eva_path + seperator + 'results.txt', 'a') as res_file:
            res_file.write(f'chain length: {DataObj.length}\n')
            res_file.write(f'Interconnection density: {DataObj.f}\n')
            res_file.write('Relative mean reconstruction error:\n')
            res_file.write(f'{round(DataObj.mre,6)}\n')
            res_file.write(f'Variance ratio explained by component 1:\n')
            res_file.write(f'{round(DataObj.varratio[0],6)}\n')
            res_file.write(f'Variance ratio explained by component 2:\n')
            res_file.write(f'{round(DataObj.varratio[1],6)}\n')
            res_file.write('-'*79 + '\n\n\n')

        # print separator line to indicate end of one condition evaluation in 
        # terminal for debugging
        print('-'*79)            
        return DataObj
    
    else:
        return None

    
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
    # calculate mean of coordinates in principal component space
    rms_c1 = np.sqrt(np.mean(DataObj.PCspaceS[:,0]**2))
    rms_c2 = np.sqrt(np.mean(DataObj.PCspaceS[:,1]**2))

    # important parameters for plotting principle components in q-space
    qmin = 1.05*np.min(DataObj.q) - 0.05*np.max(DataObj.q)
    qmax = 1.05*np.max(DataObj.q) - 0.05*np.min(DataObj.q)
    Smin = 1.05*np.min([rms_c1*DataObj.comps[0,:]*DataObj.q**2,\
                        rms_c2*DataObj.comps[1,:]*DataObj.q**2])\
            - 0.05*np.max([rms_c1*DataObj.comps[0,:]*DataObj.q**2,\
                            rms_c2*DataObj.comps[1,:]*DataObj.q**2])
    Smax = 1.05*np.max([rms_c1*DataObj.comps[0,:]*DataObj.q**2,\
                        rms_c2*DataObj.comps[1,:]*DataObj.q**2])\
            - 0.05*np.min([rms_c1*DataObj.comps[0,:]*DataObj.q**2,\
                           rms_c2*DataObj.comps[1,:]*DataObj.q**2])
    
    
    # plot principle components in q-space
    fig = plt.figure(figsize=(6, 4))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot
    ax.axis([qmin, qmax, Smin, Smax])           # set axis limits

    ax.set_title(r'Scaled principle components in $q$-space')
    ax.set_xlabel(r'$q$ / [Å$^{-1}$]')	
    ax.set_ylabel(r'$S_1(q)q^2$')

    ax.plot(DataObj.q, rms_c1*DataObj.comps[0,:]*DataObj.q**2, lw=1.0,\
            color='aqua', label='Component 1')
    ax.plot(DataObj.q, rms_c2*DataObj.comps[1,:]*DataObj.q**2, lw=1.0,\
            color='springgreen', label='Component 2')
    
    ax.legend(loc='upper right')


    if system == 'windows':
        seperator = '\\'                    # define seperator for windows 
                                            # operating system
    elif system == 'linux':
        seperator = '/'                     # define seperator for linux
                                            # operating system

    # define path to safe plot
    path = eva_path + seperator +'plots' + seperator + 'PC_in_qspace'\
           + seperator + f'N_{DataObj.length}'

    # save and close figure
    save_plot(fig=fig, name=DataObj.condi, path=path, system=system)


def plot_data_PCspace(DataObj:FileData, eva_path:str, system:str) -> None:
    '''
    Function to make script more clear. It contains all lines regarding
    the plot of the structural factor in PC-space.
    '''
    # important parameters for plotting structural factor in PC-space
    c1min = 1.05*np.min(DataObj.PCspaceS[:,0])\
            - 0.05*np.max(DataObj.PCspaceS[:,0])
    c1max = 1.05*np.max(DataObj.PCspaceS[:,0])\
            - 0.05*np.min(DataObj.PCspaceS[:,0])
    c2min = 1.05*np.min(DataObj.PCspaceS[:,1])\
            - 0.05*np.max(DataObj.PCspaceS[:,1])
    c2max = 1.05*np.max(DataObj.PCspaceS[:,1])\
            - 0.05*np.min(DataObj.PCspaceS[:,1])


    # plot structurial factor in PC-space
    fig = plt.figure(figsize=(6, 4))                # create figure
    ax = fig.add_subplot(1, 1, 1)                   # add subplot
    ax.axis([c1min, c1max, c2min, c2max])           # set axis limits

    ax.set_title(r'Structural factor in PC-space')
    ax.set_xlabel(r'Principle omponent 1')
    ax.set_ylabel(r'Principle component 2')

    ax.scatter(DataObj.PCspaceS[:,0], DataObj.PCspaceS[:,1],\
                s=0.5, c='dodgerblue')
    

    if system == 'windows':
        seperator = '\\'                    # define seperator for windows 
                                            # operating system
    elif system == 'linux':
        seperator = '/'                     # define seperator for linux
                                            # operating system

    # define path to safe plot
    path = eva_path + seperator +'plots' + seperator + 'transform_in_PCspace'\
            + seperator + f'N_{DataObj.length}'

    # save and close figure
    save_plot(fig=fig, name=DataObj.condi, path=path, system=system)


def plot_data_qspace(DataObj:FileData, eva_path:str, system:str) -> None:
    '''
    Function to make script more clear. It contains all lines regarding
    the plot of the example curves in q-space.
    '''
    # important parameters for plotting example curves and mean curve in
    # q-space
    qmin = 1.05*np.min(DataObj.q) - 0.05*np.max(DataObj.q)
    qmax = 1.05*np.max(DataObj.q) - 0.05*np.min(DataObj.q)
    Smin = 1.05*np.min([DataObj.mS*DataObj.q**2,\
                        DataObj.S[50,:]*DataObj.q**2,\
                        DataObj.S[350,:]*DataObj.q**2,\
                        DataObj.reconS[50,:]*DataObj.q**2,\
                        DataObj.reconS[350,:]*DataObj.q**2])\
           - 0.05*np.max([DataObj.mS*DataObj.q**2,\
                          DataObj.S[50,:]*DataObj.q**2,\
                          DataObj.S[350,:]*DataObj.q**2,\
                          DataObj.reconS[50,:]*DataObj.q**2,\
                          DataObj.reconS[350,:]*DataObj.q**2])
    Smax = 1.05*np.max([DataObj.mS*DataObj.q**2,\
                        DataObj.S[50,:]*DataObj.q**2,\
                        DataObj.S[350,:]*DataObj.q**2,\
                        DataObj.reconS[50,:]*DataObj.q**2,\
                        DataObj.reconS[350,:]*DataObj.q**2])\
           - 0.05*np.min([DataObj.mS*DataObj.q**2,\
                          DataObj.S[50,:]*DataObj.q**2,\
                          DataObj.S[350,:]*DataObj.q**2,\
                          DataObj.reconS[50,:]*DataObj.q**2,\
                          DataObj.reconS[350,:]*DataObj.q**2])
    
    
    # plot example curves and mean curve in q-space
    fig = plt.figure(figsize=(8, 6))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot
    #ax.axis([qmin, qmax, Smin, Smax])           # set axis limits

    ax.set_title(r'Experimental, reconstructed and mean structural factor in $q$-space')
    ax.set_xlabel(r'$q$ / [Å$^{-1}$]')
    ax.set_ylabel(r'$S(q)q^2$')

    ax.loglog(DataObj.q, DataObj.S[50,:]*DataObj.q**2, lw=1.0,\
              color='dodgerblue', label='example curve 1')
    ax.loglog(DataObj.q, DataObj.S[350,:]*DataObj.q**2, lw=1.0,\
              color='lightskyblue', label='example curve 2')
    ax.loglog(DataObj.q, DataObj.reconS[50,:]*DataObj.q**2, lw=1.0,\
              ls='--', color='dodgerblue',\
              label='reconstructed example curve 1')
    ax.loglog(DataObj.q, DataObj.reconS[350,:]*DataObj.q**2, lw=1.0,\
              ls='--', color='lightskyblue',\
              label='reconstructed example curve 2')
    ax.loglog(DataObj.q, DataObj.rmsS*DataObj.q**2, lw=1.0,\
              color='black', label='mean curve')
    
    ax.legend(loc='lower right')            # set legend position


    if system == 'windows':
        seperator = '\\'                    # define seperator for windows 
                                            # operating system
    elif system == 'linux':
        seperator = '/'                     # define seperator for linux
                                            # operating system

    # define path to safe plot
    path = eva_path + seperator +'plots' + seperator\
           + 'example_in_qspace' + seperator + f'N_{DataObj.length}'

    # save and close figure
    save_plot(fig=fig, name=DataObj.condi, path=path, system=system)


def plot_recon_error(DataObj:FileData, eva_path:str, system:str) -> None:
    '''
    Function to make script more clear. It contains all lines regarding
    the plot of the reconstruction error, mean curves and example curves in
    q-space.
    '''
    # important parameters for plotting reconstruction error in q-space
    qmin = 1.05*np.min(DataObj.q) - 0.05*np.max(DataObj.q)
    qmax = 1.05*np.max(DataObj.q) - 0.05*np.min(DataObj.q)
    Smin = 1.05*np.min(DataObj.re) - 0.05*np.max(DataObj.re)
    Smax = 1.05*np.max(DataObj.re) - 0.05*np.min(DataObj.re)

    
    # plot reconstruction error in q-space
    fig = plt.figure(figsize=(8, 6))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot
    #ax.axis([qmin, qmax, Smin, Smax])           # set axis limits

    ax.set_title(r'Relative reconstruction error in $q$-space')
    ax.set_xlabel(r'$q$ / [Å$^{-1}$]')
    ax.set_ylabel(r'$e_S$')
    
    ax.loglog(DataObj.q, DataObj.re, lw=1.0, color='tomato')
    

    if system == 'windows':
        seperator = '\\'                    # define seperator for windows 
                                            # operating system
    elif system == 'linux':
        seperator = '/'                     # define seperator for linux
                                            # operating system

    # define path to safe plot
    path = eva_path + seperator +'plots' + seperator\
           + 'eS_in_qspace' + seperator + f'N_{DataObj.length}'

    # save and close figure
    save_plot(fig=fig, name=DataObj.condi, path=path, system=system)