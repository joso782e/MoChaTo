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
import json
import h5py
import numpy as np
from sklearn.decomposition import PCA
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt


# print import statement if the module has been imported successfully
print("Module \"library_MoChaTo.py\" imported successfully")


with open('.\\Scripts\\config.json', 'r') as configf:
    config = json.load(configf)


class FileData(PCA):
    '''
    Class to store and update results for comprehensive file evaluation after 
    individual file evaluation. It inherits from the PCA class of sklearn and contains the following attributes:
    - condi:        string describing simulated condition
    - length:       chainlength in number of spheres
    - f:            connector distance
    - q:            values in q-space
    - S:            structural factor in q-space
    - clmat:        crosslink matrix
    - mS:           mean of structural factor in q-space
    - ll:           loop length
    - mll:          mean loop length
    - PCspaceS:     structural factor in PC-space
    - reconS:       reconstructed structural factor in q-space
    - mre:          relative mean reconstruction error
    - re:           relative reconstruction error in q-space
    '''   
    # define contructor
    def __init__(
        self,
        condi:str,
        length:int,
        f:int,
        q:np.ndarray,
        S:np.ndarray,
        clmat:np.ndarray,
        n_components=None,
        *,
        copy=True,
        whiten=False,
        svd_solver="auto",
        tol=0.0,
        iterated_power="auto",
        n_oversamples=10,
        power_iteration_normalizer="auto",
        random_state=None,
    ):
        super().__init__(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            n_oversamples=n_oversamples,
            power_iteration_normalizer=power_iteration_normalizer,
            random_state=random_state,
        )
        self.condi = condi
        self.length = length
        self.f = f
        self.q = q
        self.S = S
        self.clmat = clmat
        self.mS = np.mean(S, axis=0)
        self.ll = np.abs(self.clmat[:,:,1] - self.clmat[:,:,2])
        self.mll = np.mean(self.ll, axis=1)
        self.fit(self.S)
        self.PCspaceS = self.transform(self.S)
        self.per_recon()

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

        return FileData(n_components=ncomps, condi=Cond, length=NChain,\
                        f=InterconDens, q=qKey, S=SData,\
                        clmat=crosslinks)
    
    def per_recon(self) -> None:
        '''
        Function to calculate reconstructed date and related quantities from
        PCA
        '''
        # calculate reconstructed data
        reconS = np.matmul(self.PCspaceS, self.components_) + self.mS

        # calculate difference between original and reconstructed data
        diff = self.S - reconS

        # calculate relative mean reonstruction error
        mre = np.sqrt(np.mean((diff)**2))/np.mean(self.mS)
        # variance of error
        mrevar = np.mean((diff - mre*np.mean(self.mS))**2)

        # calculate relative reconstruction error in q-space
        re = np.sqrt(np.mean((diff)**2, axis=0))/self.mS
        # variance of error
        revar = np.mean((diff - re*self.mS)**2, axis=0)

        setattr(self, 'reconS', reconS)     # set reconstructed data
        setattr(self, 'mre', mre)           # set relative mean error
        setattr(self, 're', re)             # set relative reconstruction error
        setattr(self, 'mrevar', mrevar)     # set variance of relative mean
                                            # error
        setattr(self, 'revar', revar)       # set variance of relative error


def filter_func(name:str, file:h5py.File, NComps:int=config['NComps'],\
                filter_obj:str=config['filter_obj'],\
                eva_path:str=config['eva_path'], system:str=config['system'])\
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

        # print and save results in seperate .txt file
        with open(eva_path + seperator + 'results.txt', 'a') as res_file:
            res_file.write(f'chain length:              {DataObj.length}\n' +\
                           f'Connector distance:        {DataObj.f}\n' +\
                           r'$\langle e_S\rangle$:     ' +\
                           f'{round(DataObj.mre,6)}\n' +\
                           r'$\sigma_1$:               ' +\
                           f'{round(DataObj.explained_variance_ratio_[0],6)}'\
                    '\n' + r'$\sigma_2$:               ' +\
                           f'{round(DataObj.explained_variance_ratio_[1],6)}')
            res_file.write('-'*79 + '\n\n\n')

        # print separator line to indicate end of one condition evaluation in 
        # terminal for debugging
        print('-'*79)            
        return DataObj
    
    else:
        return None

    
def save_plot(fig:plt.Figure, name:str, path:str, system:str=config['system'],\
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


def plot_princ_comps(DataObj:FileData, eva_path:str=config['eva_path'],\
                     system:str=config['system']) -> None:
    '''
    Function to make script more clear. It contains all lines regarding
    the plot of the principle components.
    '''
    # calculate mean of coordinates in principal component space
    rms_c1 = np.sqrt(np.mean(DataObj.PCspaceS[:,0]**2))
    rms_c2 = np.sqrt(np.mean(DataObj.PCspaceS[:,1]**2))

    # important parameters for plotting principle components in q-space
    qmin = np.exp(1e-2)
    qmax = 1
    Smin = 1.05*np.min([rms_c1*DataObj.components_[0,:]*DataObj.q**2,\
                        rms_c2*DataObj.components_[1,:]*DataObj.q**2])\
            - 0.05*np.max([rms_c1*DataObj.components_[0,:]*DataObj.q**2,\
                            rms_c2*DataObj.components_[1,:]*DataObj.q**2])
    Smax = 1.05*np.max([rms_c1*DataObj.components_[0,:]*DataObj.q**2,\
                        rms_c2*DataObj.components_[1,:]*DataObj.q**2])\
            - 0.05*np.min([rms_c1*DataObj.components_[0,:]*DataObj.q**2,\
                           rms_c2*DataObj.components_[1,:]*DataObj.q**2])
    
    # plot principle components in q-space
    fig = plt.figure(figsize=(6, 4))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot
    #ax.axis([qmin, qmax, Smin, Smax])           # set axis limits

    ax.set_title(r'Scaled principle components in $q$-space')
    ax.set_xlabel(r'$q$ / [Å$^{-1}$]')	
    ax.set_ylabel(r'$S_1(q)q^2$')

    ax.loglog(DataObj.q, rms_c1*DataObj.components_[0,:]*DataObj.q**2, lw=1.0,\
            color='aqua', label='Component 1')
    ax.loglog(DataObj.q, rms_c2*DataObj.components_[1,:]*DataObj.q**2, lw=1.0,\
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
    save_plot(fig=fig, name=DataObj.condi, path=path)


def plot_data_PCspace(DataObj:FileData, eva_path:str=config['eva_path'],\
                     system:str=config['system']) -> None:
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
    
    color = DataObj.mll/np.max(DataObj.mll)         # color for scatter plot

    # plot structurial factor in PC-space
    fig = plt.figure(figsize=(6, 4))                # create figure
    ax = fig.add_subplot(1, 1, 1)                   # add subplot
    ax.axis([c1min, c1max, c2min, c2max])           # set axis limits

    ax.set_title(r'Structural factor in PC-space')
    ax.set_xlabel(r'Principle omponent 1')
    ax.set_ylabel(r'Principle component 2')

    scatter = ax.scatter(DataObj.PCspaceS[:,0], DataObj.PCspaceS[:,1], s=0.5,\
                         c=color, cmap='RdBu')
    
    axcb = fig.colorbar(scatter, ax=ax)
    axcb.set_label('mean loop length')
    
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
    save_plot(fig=fig, name=DataObj.condi, path=path)


def plot_data_qspace(DataObj:FileData, eva_path:str=config['eva_path'],\
                     system:str=config['system']) -> None:
    '''
    Function to make script more clear. It contains all lines regarding
    the plot of the example curves in q-space.
    '''
    # important parameters for plotting example curves and mean curve in
    # q-space
    qmin = -2
    qmax = 1.0
    Smin = np.exp(1e-4)
    Smax = np.max(DataObj.S)*1.05
    
    # plot example curves and mean curve in q-space
    fig = plt.figure(figsize=(8, 6))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot
    ax.axis([qmin, qmax, Smin, Smax])           # set axis limits

    ax.set_title('Experimental, reconstructed and mean structural factor in'\
                 + r' $q$-space')
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
    ax.loglog(DataObj.q, DataObj.mS*DataObj.q**2, lw=1.0,\
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
    save_plot(fig=fig, name=DataObj.condi, path=path)


def plot_recon_error(DataObj:FileData, eva_path:str=config['eva_path'],\
                     system:str=config['system']) -> None:
    '''
    Function to make script more clear. It contains all lines regarding
    the plot of the reconstruction error, mean curves and example curves in
    q-space.
    '''
    # important parameters for plotting reconstruction error in q-space
    qmin = np.exp(1e-2)
    qmax = 1.0
    Smin = np.exp(1e-4)
    Smax = np.max(DataObj.re)*1.05

    # plot reconstruction error in q-space
    fig = plt.figure(figsize=(8, 6))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot
    ax.axis([qmin, qmax, Smin, Smax])           # set axis limits

    ax.set_title(r'Relative reconstruction error in $q$-space')
    ax.set_xlabel(r'$q$ / [Å$^{-1}$]')
    ax.set_ylabel(r'$e_S$')
    
    ax.errorbar(DataObj.q, DataObj.re, yerr=np.sqrt(DataObj.revar),  lw=1.0,\
                color='tomato')
    
    ax.set_yscale('log')                     # set y-axis to log scale
    ax.set_xscale('log')                     # set x-axis to log scale
    
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
    save_plot(fig=fig, name=DataObj.condi, path=path)


def plot_ll_hist(DataObj:FileData, binnum:int=config['binnum'],\
                   eva_path:str=config['eva_path'],\
                   system:str=config['system']) -> None:
    '''
    Function to make script more clear. It contains all lines regarding
    the plot of the loop length histogram. The bins are normalized as
    such, that the area under the histogramm integrates to 1. For further
    information please refere to the documentation of plt.hist().
    '''
    # important parameters for plotting loop length histogram
    bins = np.linspace(np.min(DataObj.ll), np.max(DataObj.ll), binnum)
    llflat = DataObj.ll.flatten()

    # plot loop length histogram
    fig = plt.figure(figsize=(8, 6))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot

    ax.set_title(r'Histogram of loop lengths')
    ax.set_xlabel(r'monomers per loop')
    ax.set_ylabel(r'relative frequency')

    ax.hist(llflat, bins=bins, density=True, color='dodgerblue')

    if system == 'windows':
        seperator = '\\'                    # define seperator for windows 
                                            # operating system
    elif system == 'linux':
        seperator = '/'                     # define seperator for linux
                                            # operating system

    # define path to safe plot
    path = eva_path + seperator +'plots' + seperator\
           + 'll_hist' + seperator + f'N_{DataObj.length}'

    # save and close figure
    save_plot(fig=fig, name=DataObj.condi, path=path)


def plot_mll_hist(DataObj:FileData, binnum:int=config['binnum'],\
                   eva_path:str=config['eva_path'],\
                   system:str=config['system']) -> None:
    '''
    Function to make script more clear. It contains all lines regarding
    the plot of the mean loop length histogram. The bins are normalized as
    such, that the area under the histogramm integrates to 1. For further
    information please refere to the documentation of plt.hist().
    '''
    # important parameters for plotting loop length histogram
    bins = np.linspace(np.min(DataObj.mll), np.max(DataObj.mll), binnum)

    # plot loop length histogram
    fig = plt.figure(figsize=(8, 6))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot

    ax.set_title(r'Histogram of mean loop lengths')
    ax.set_xlabel(r'mean number of monomers per loop')
    ax.set_ylabel(r'relative frequency')

    ax.hist(DataObj.mll, bins=bins, density=False, color='dodgerblue')

    if system == 'windows':
        seperator = '\\'                    # define seperator for windows 
                                            # operating system
    elif system == 'linux':
        seperator = '/'                     # define seperator for linux
                                            # operating system

    # define path to safe plot
    path = eva_path + seperator +'plots' + seperator\
           + 'mll_hist' + seperator + f'N_{DataObj.length}'

    # save and close figure
    save_plot(fig=fig, name=DataObj.condi, path=path)