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
import scipy.optimize as opt
from sklearn.decomposition import PCA
from matplotlib import colormaps as cm
import matplotlib.pyplot as plt


# print import statement if the module has been imported successfully
print("Module \"library_MoChaTo.py\" imported successfully")


with open('.\\data_evaluation\\Scripts\\config.json', 'r') as configf:
    config = json.load(configf)


class FileData(PCA):
    '''
    Class to store and update results for comprehensive file evaluation after 
    individual file evaluation. It inherits from the PCA class of sklearn and contains the following attributes:
    - condi:        string describing simulated condition
    - length:       chainlength in number of spheres
    - f:            connector distance
    - q:            values in q-space
    - S:            form factor in q-space
    - clmat:        crosslink matrix
    - mS:           mean of form factor in q-space
    - empvar:       empirical variance of form factor in q-space
    - ll:           loop length
    - mll:          mean loop length
    - PCspaceS:     form factor in PC-space
    - reconS:       reconstructed form factor in q-space
    - mre:          relative mean reconstruction error
    - re:           relative reconstruction error in q-space
    - mrevar:       variance of relative mean reconstruction error
    - revar:        variance of relative reconstruction error in q-space
    '''   
    # define contructor
    def __init__(
        self,
        condi:str,
        N:int,
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
        self.N = N
        self.f = 1/f
        self.q = q
        self.S = S
        self.clmat = np.asanyarray(clmat, dtype=np.int_)
        self.mS = np.mean(S, axis=0)
        self.empvar = np.var(S, axis=0)/self.mS**2
        self.LLCalc()
        self.fit(self.S)
        self.PCspaceS = self.transform(self.S)
        self.Per_Recon()
        self.c1 = self.PCspaceS[:,0]
        self.c2 = self.PCspaceS[:,1]
        self.PC1 = self.components_[0,:]
        self.PC2 = self.components_[1,:]

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

        return FileData(n_components=ncomps, condi=Cond, N=NChain,\
                        f=InterconDens, q=qKey, S=SData,\
                        clmat=crosslinks)
    
    def Per_Recon(self) -> None:
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
        mrevar = np.mean((diff/np.mean(self.mS) - mre)**2)

        # calculate relative reconstruction error in q-space
        re = np.sqrt(np.mean((diff)**2, axis=0))/self.mS
        # variance of error
        revar = np.mean((diff/self.mS - re)**2, axis=0)

        setattr(self, 'reconS', reconS)     # set reconstructed data
        setattr(self, 'mre', mre)           # set relative mean error
        setattr(self, 're', re)             # set relative reconstruction error
        setattr(self, 'mrevar', mrevar)     # set variance of relative mean
                                            # error
        setattr(self, 'revar', revar)       # set variance of relative error
    
    def LLCalc(self) -> None:
        '''
        Function to calculate loop length an mean loop length from crosslink
        matrix
        '''
        ll = np.abs(self.clmat[:,:,1] - self.clmat[:,:,2])
        ll = np.split(ll, indices_or_sections=ll.shape[0], axis=0)
        self.ll = [l[np.nonzero(l)] for l in ll]
        self.mll = ([np.mean(l) if len(l) > 0 else 0 for l in self.ll])
        self.nl = [len(l) for l in self.ll]
    
    def PerfFit(self, FitFunc, xdata, ydata, fitname) -> None:
        '''
        Funktion to calculate quantities from fit to data
        updates self with new attributes as follows:
        - self.{fitname}{i} fitted values
        - self.{fitname}err{i} uncertainty of fitted values
        for i = 0, ..., n-1  fit parameters
        '''
        xdata = getattr(self, xdata)
        ydata = getattr(self, ydata)

        for i in range(ydata.shape[0]):
            fit, _ = opt.curve_fit(
                FitFunc, xdata=xdata, ydata=ydata[i,:]
            )

            if i == 0:
                for j in range(len(fit)):
                    
                    setattr(self, f'{fitname}{j}', [fit[j]])
            else:
                for j in range(len(fit)):
                    f = getattr(self, f'{fitname}{j}')

                    f.append(fit[j])

                    setattr(self, f'{fitname}{j}', f)


class Cycler:
    '''
    Class to create a cycler
    '''

    def __init__(self, iterable):
        self.count = 0
        self.iterable = iterable
    
    def Cycle(self):
        '''
        Function to cycle through iterable
        '''
        if self.count >= len(self.iterable):
            self.count = 0
        item = self.iterable[self.count]
        self.count += 1
        return item
    
    def FullCycle(self):
        '''
        Function to perform full cycle through iterable
        '''
        for i in range(len(self.iterable)):
            yield self.Cycle()


class PlotRule:
    '''
    Class to create plot rules 
    '''

    def __init__(
            self,
            plotaspects:dict
            ):
        self.Nrule = 40
        self.frule = 1/23
        self.binnum = 30
        self.figsize = (8, 6)
        self.title = 'Plot title'
        self.xlabel = 'xlable'
        self.ylabel = 'ylable'
        self.xdata = 'q'
        self.ydata = 'S'
        self.xlim = [None, None]
        self.ylim = [None, None]
        self.xscale = 'linear'
        self.yscale = 'linear'
        self.scalfac = 1.0
        self.lw = 1.0
        self.ls = '-'
        self.color = 'dodgerblue'
        self.marker='o'
        self.ms = 3.5
        self.plotdomain = 'Kratky'
        self.plot = 'diag'
        self.sortby = 'N'
        self.legend = True
        self.legend_loc = 'upper right'
        self.label = 'legend label'

        # update attributes with values from plotaspects and aspectvalues
        for i in plotaspects.keys():
            setattr(self, f'{i}', plotaspects[i])


class PlotData:

    def __init__(
            self,
            plotaspects:list,
            dataobjs:list[FileData]
    ):
        self.rule = PlotRule(plotaspects=plotaspects)

        if not (
            (self.rule.Nrule in [obj.N for obj in dataobjs]) or
            (self.rule.Nrule != 'all')
            ):
            raise ValueError(
                f'Rule "N = {self.rule.Nrule}" not in data objects. '
                 'Please set Nrule to one or a list of the following values:'
                f'\n{np.unique([obj.N for obj in dataobjs])}'
                 '\nor to "all"'
            )
        
        if not (
            (self.rule.frule in [obj.f for obj in dataobjs]) or
            (self.rule.frule != 'all')
        ):
            raise ValueError(
                f'Rule "f = {self.rule.frule}" not in data objects. '
                 'Please set frule to one or a list of one over the following '
                 'values:'
                f'\n{np.unique([int(1/obj.f) for obj in dataobjs])}'
                 '\nor to "all"'
            )
        
        if not (
            self.rule.xdata in dir(dataobjs[0]) or
            self.rule.xdata in ['c1', 'c2', 'PC1', 'PC2']
            ):
            raise AttributeError(
                f'Rule "xdata = {self.rule.xdata}" not as attribute in '
                 'FileData objects. Please set xdata to one of the following '
                 'values:'
                f'\n{dir(dataobjs[0])}'
                '\n,to "c1"/"c2" for coordinate 1/2 in PC-space '
                'or to "PC1"/"PC2" for principal component 1/2'
            )
        
        if not (
            self.rule.ydata in dir(dataobjs[0]) or
            self.rule.ydata in ['c1', 'c2', 'PC1', 'PC2']
            ):
            raise AttributeError(
                f'Rule "ydata = {self.rule.ydata}" not as attribute in '
                 'FileData objects. Please set ydata to one of the following '
                 'values:'
                f'\n{dir(dataobjs[0])}'
                '\n, to "c1"/"c2" for coordinate 1/2 in PC-space '
                'or to "PC1"/"PC2" for principal component 1/2'
            )
        
        if not (self.rule.sortby == 'N' or self.rule.sortby == 'f'):
            raise ValueError(
                f'Input "{self.rule.sortby}" for sortby not supported. '
                'Please set sortby to "N" or "f"'
            )        

        self.SetData(dataobjs=dataobjs)
    
    def SetData(self, dataobjs:list[FileData]) -> None:
        '''
        Function to update data attributes in PlotData object
        '''
        rule = self.rule

        if rule.Nrule == 'all':
            rule.Nrule = np.unique([obj.N for obj in dataobjs])

        if rule.frule == 'all':
            rule.frule = np.unique([obj.f for obj in dataobjs])

        data = [
            obj for obj in dataobjs
            if (obj.N in rule.Nrule) and (obj.f in rule.frule)
        ]

        self.data = data

    def CreatePlot(
            self, eva_path:str=config['eva_path'], system:str=config['system']
    ) -> None:
        '''
        Function to create and save plot with data from FileData objects in
        PlotData object
        '''
        rule = self.rule

        if system == 'windows':
            seperator = '\\'            # define seperator for windows 
                                        # operating system
        elif system == 'linux':
            seperator = '/'             # define seperator for linux
                                        # operating system
        else:
            raise ValueError(
                f'Input "{system}" for system not supported. '
                'Please set system to "windows" or "linux"'
            )        
        figpath = eva_path + seperator + 'plots' + seperator +\
            rule.plot + '_' + rule.plotdomain + '_' + rule.ydata +\
            '_vs_'+ rule.xdata + seperator + f'N_{rule.Nrule}'
                
        fig = plt.figure(figsize=rule.figsize)      # create figure
        ax = fig.add_subplot(1, 1, 1)               # create subplot

        # set title and axis labels
        ax.set_title(rule.title)
        ax.set_xlabel(rule.xlabel)
        ax.set_ylabel(rule.ylabel)
        ax.grid(axis='both', ls='--', lw=0.5, color='grey')

        xdata = []
        ydata = []

        for i in getattr(rule, f'{rule.sortby}rule'):
            if len([
                obj for obj in self.data
                if getattr(obj, rule.sortby) == i
            ]) == 0:
                continue
            xdata.append(
                np.stack([
                    getattr(obj, rule.xdata) for obj in self.data
                    if getattr(obj, rule.sortby) == i
                ], axis=0)
            )
            ydata.append(
                np.stack([
                    getattr(obj, rule.ydata) for obj in self.data
                    if getattr(obj, rule.sortby) == i
                ], axis=0)
            )

        ax.set_prop_cycle(
            color=rule.color,
            ls=rule.ls,
            lw=rule.lw,
            marker=rule.marker,
            ms=rule.ms,
        )

        for i in range(len(xdata)):
            # asume more data points then samples
            if xdata[i].shape[0] < xdata[i].shape[1]:
                xdata[i] = xdata[i].T
            # check if one axis lenght of ydata[i] matches length of axis 0 in
            # xdata[i], if not raise ValueError
            if xdata[i].shape[0] != ydata[i].shape[0]:
                ydata[i] = ydata[i].T
                if xdata[i].shape[0] != ydata[i].shape[0]:
                    raise ValueError(
                        f'Data set {i+1} can not be plotted against each '
                        f'other because their shapes {xdata[i].shape} and '
                        f'{ydata[i].shape} can not be broadcast together.'
                    )
            
            
            
            # plot data as diagram or histogram
            if rule.plot == 'diag':
                # dicide wether to turn auto scale on or not
                xup = rule.xlim[0]
                xlow = rule.xlim[1]
                yup = rule.ylim[0]
                ylow = rule.ylim[1]
                if xup and xlow and (type(xup), type(xlow)) != (float, float):
                    xauto = False
                else:
                    xauto = True
                
                if yup and ylow and (type(yup), type(ylow)) != (float, float):
                    yauto = False
                else:
                    yauto = True

                # set axis view limits
                ax.set_xlim(left=xlow, right=xup, auto=xauto)
                ax.set_ylim(bottom=ylow, top=yup, auto=yauto)

                if rule.plotdomain == 'Kratky':
                    # plot data in Kratky plot
                    ydata[i] = ydata[i]*xdata[i]**2
                    ax.loglog(
                        xdata[i], rule.scalfac*ydata[i], color=rule.color[i],
                        linestyle=rule.ls[i], lw=rule.lw[i],
                        marker=rule.marker[i], ms=rule.ms[i],
                        label=rule.label[i]
                    )
                else:
                    # plot data in normal plot
                    ax.plot(
                        xdata[i], rule.scalfac*ydata[i],
                        label=rule.label[i]
                    )
                    ax.set_xscale(rule.xscale)
                    ax.set_yscale(rule.yscale)

            elif rule.plot == 'hist':
                # create bins and plot data in histogram
                bins = np.linspace(
                    np.min(ydata[i]), np.max(ydata[i]), rule.binnum
                )
                ax.hist(
                    ydata[i].flatten(), bins=bins, density=True,
                    color=rule.color[i], label=rule.label
                )
                # display mean value and calculate optimal position for text
                # and line
                hax = ax.get_ylim()[1] - ax.get_ylim()[0]
                ymin = -ax.get_ylim()[0]/hax
                ymax = ax.get_ylim()[1]/hax
                ax.axvline(
                    np.mean([ydata[i]]), ymin, ymax, 
                )
                ax.text(np.mean(
                    ydata[i]), ymax*hax,
                    f'mean value:\n{round(np.mean(ydata[i]),4)}',
                    horizontalalignment='center',
                    verticalalignment='bottom'
                )
            else:
                raise ValueError(
                    f'Plot type "{rule.plot}" not supported. '
                    'Please set plot to "diag" or "hist"'
                )
            
        if rule.legend:
            ax.legend(loc=rule.legend_loc)
        
        save_plot(
            fig=fig, name=f'f_{[round(f,2) for f in rule.frule]}', path=figpath
        )

        
def filter_func(
        name:str, file:h5py.File, NComps:int=config['NComps'],
        filter_obj:str=config['filter_obj'], eva_path:str=config['eva_path'], 
        system:str=config['system']
) -> None:
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
            res_file.write(f'chain length:              {DataObj.N}\n' +\
                           f'Connector distance:        {DataObj.f}\n' +\
                            'empirical variance:         ' +\
                           f'{round(np.mean(DataObj.empvar),6)}\n' +\
                           r'$\langle e_S\rangle$:      ' +\
                           f'{round(DataObj.mre,6)}\n' +\
                           r'$\sigma_1$:                ' +\
                           f'{round(DataObj.explained_variance_ratio_[0],6)}'\
                    '\n' + r'$\sigma_2$:                ' +\
                           f'{round(DataObj.explained_variance_ratio_[1],6)}')
            res_file.write('\n' + '-'*79 + '\n\n\n')

        # print separator line to indicate end of one condition evaluation in 
        # terminal for debugging
        print('-'*79)            
        return DataObj
    
    else:
        return None

    
def save_plot(
        fig:plt.Figure, name:str, path:str, system:str=config['system'],
        fileformat:str=config['fileformat']
) -> None:
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
    qmin = 1e-2

    
    # plot principle components in q-space
    fig = plt.figure(figsize=(8, 6))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot

    ax.set_title(r'Scaled principle components in $q$-space')
    ax.set_xlabel(r'$q$ / [Å$^{-1}$]')	
    ax.set_ylabel(r'$S_1(q)q^2$')

    ax.loglog(DataObj.q, rms_c1*DataObj.components_[0,:]*DataObj.q**2, lw=1.0,\
            color='aqua', label='Component 1')
    ax.loglog(DataObj.q, rms_c2*DataObj.components_[1,:]*DataObj.q**2, lw=1.0,\
            color='springgreen', label='Component 2')
    
    ax.legend(loc='upper right')
    ax.set_xlim(left=qmin)                  # set x-axis limits


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
    the plot of the form factor in PC-space.
    '''
    # important parameters for plotting form factor in PC-space
    c1min = 1.05*np.min(DataObj.PCspaceS[:,0])\
            - 0.05*np.max(DataObj.PCspaceS[:,0])
    c1max = 1.05*np.max(DataObj.PCspaceS[:,0])\
            - 0.05*np.min(DataObj.PCspaceS[:,0])
    c2min = 1.05*np.min(DataObj.PCspaceS[:,1])\
            - 0.05*np.max(DataObj.PCspaceS[:,1])
    c2max = 1.05*np.max(DataObj.PCspaceS[:,1])\
            - 0.05*np.min(DataObj.PCspaceS[:,1])
    
    color = DataObj.mll/np.max(DataObj.mll)         # color for scatter plot

    # plot form factor in PC-space
    fig = plt.figure(figsize=(8, 6))                # create figure
    ax = fig.add_subplot(1, 1, 1)                   # add subplot
    ax.axis([c1min, c1max, c2min, c2max])           # set axis limits

    ax.set_title(r'Form factor in PC-space')
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
    qmin = 1e-2
    Smin = 1e-4
    
    # plot example curves and mean curve in q-space
    fig = plt.figure(figsize=(8, 6))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot

    ax.set_title('Experimental, reconstructed and mean form factor in'\
                 + r' $q$-space')
    ax.set_xlabel(r'$q$ / [Å$^{-1}$]')
    ax.set_ylabel(r'$S(q)q^2$')

    ax.plot(DataObj.q, DataObj.S[50,:]*DataObj.q**2, lw=1.0,\
            color='dodgerblue', label='example curve 1')
    ax.plot(DataObj.q, DataObj.S[350,:]*DataObj.q**2, lw=1.0,\
            color='lightskyblue', label='example curve 2')
    ax.plot(DataObj.q, DataObj.reconS[50,:]*DataObj.q**2, lw=1.0,\
            ls='--', color='dodgerblue',\
            label='reconstructed example curve 1')
    ax.plot(DataObj.q, DataObj.reconS[350,:]*DataObj.q**2, lw=1.0,\
            ls='--', color='lightskyblue',\
            label='reconstructed example curve 2')
    ax.plot(DataObj.q, DataObj.mS*DataObj.q**2, lw=1.0, color='black',\
            label='mean curve with empirical variance')
    
    ax.legend(loc='lower right')            # set legend position
    ax.set_yscale('log')                    # set y-axis to log scale
    ax.set_xscale('log')                    # set x-axis to log scale
    ax.set_xlim(left=qmin)                  # set x-axis limits
    ax.set_ylim(bottom=Smin)                # set y-axis limits


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
    qmin = 1e-2

    # plot reconstruction error in q-space
    fig = plt.figure(figsize=(8, 6))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot

    ax.set_title(r'Relative reconstruction error in $q$-space')
    ax.set_xlabel(r'$q$ / [Å$^{-1}$]')
    ax.set_ylabel(r'$e_S$')
    
    ax.plot(DataObj.q, DataObj.re*DataObj.q**2,  lw=1.0, color='tomato',\
            label=r'$e_S(q)$')
    ax.plot(DataObj.q, np.sqrt(DataObj.empvar)*DataObj.q**2, lw=1.0,\
            color='dodgerblue', label=r'$\sigma_{emp}$')
    ax.plot(DataObj.q, np.sqrt(DataObj.revar)*DataObj.q**2, lw=1.0, ls='--',\
            color='tomato', label=r'$\sigma_{e_S}$')
    
    ax.legend(loc='upper right')            # set legend position
    
    ax.set_yscale('log')                     # set y-axis to log scale
    ax.set_xscale('log')                     # set x-axis to log scale
    ax.set_xlim(left=qmin)                   # set x-axis limits
    
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


def plot_mll_vs_ci(DataObj:FileData, i:int, eva_path:str=config['eva_path'],\
                    system:str=config['system']) -> None:
    '''
    Function to make script more clear. It contains all lines regarding the
    plot of the mean loop length depending on the coordinate in the "i"-th
    principle component.
    '''
    # important parameters for plotting mean loop length depending on
    # the coordinates in the "i"-th principle component
    cimin = 1.05*np.min(DataObj.PCspaceS[:,i-1])\
           - 0.05*np.max(DataObj.PCspaceS[:,i-1])
    cimax = 1.05*np.max(DataObj.PCspaceS[:,i-1])\
           - 0.05*np.min(DataObj.PCspaceS[:,i-1])
    mllmin = 1.05*np.min(DataObj.mll)\
             - 0.05*np.max(DataObj.mll)
    mllmax = 1.05*np.max(DataObj.mll)\
             - 0.05*np.min(DataObj.mll)
    
    # plot mll vs ci
    fig = plt.figure(figsize=(8, 6))            # create figure
    ax = fig.add_subplot(1, 1, 1)               # add subplot
    ax.axis([cimin, cimax, mllmin, mllmax])     # set axis limits

    ax.set_title(r'Mean loop length depending on' + f' $c_{i}$ in PC-space')
    ax.set_xlabel(fr'$c_{i}$')
    ax.set_ylabel(r'mean number of monomers per loop')

    ax.plot(DataObj.PCspaceS[:,i-1], DataObj.mll, ls='None', marker='o',\
            ms=3.5, mfc='dodgerblue', mec='dodgerblue')


    if system == 'windows':
        seperator = '\\'                    # define seperator for windows 
                                            # operating system
    elif system == 'linux':
        seperator = '/'                     # define seperator for linux
                                            # operating system

    # define path to safe plot
    path = eva_path + seperator +'plots' + seperator\
           + f'mll_vs_c{i}' + seperator + f'N_{DataObj.length}'

    # save and close figure
    save_plot(fig=fig, name=DataObj.condi, path=path)