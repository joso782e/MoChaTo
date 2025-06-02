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
        self.EmpEva()        

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
    
    def EmpEva(self) -> None:
        '''
        Function to perform evaluation on empiric data and store results in
        self,
        updates self with new attributes as follows:
        - self.mS:          mean of form factor in q-space
        - self.emperr:      empirical error of form factor in q-space
        - self.LL:          loop length per SCNP
        - self.mll:         mean loop length per SCNP
        - self.nl:          number of loops per SCNP
        '''
        self.mS = np.mean(self.S, axis=0)
        self.emperr = np.sqrt(
            np.mean((self.S - self.mS)**2/self.mS**2, axis=0)
        )

    def LLCalc(self) -> None:
        '''
        Function to calculate loop length an mean loop length from crosslink
        matrix
        updates self with new attributes as follows:
        - self.ll:          loop length per SCNP
        - self.mll:         mean loop length per SCNP
        - self.nl:          number of loops per SCNP
        '''
        ll = np.abs(self.clmat[:,:,1] - self.clmat[:,:,2])
        ll = np.split(ll, indices_or_sections=ll.shape[0], axis=0)

        # set loop length
        self.ll = [l[np.nonzero(l)] for l in ll]
        # set mean loop length
        self.mll = ([np.mean(l) if len(l) > 0 else 0 for l in self.ll])
        # set number of loops
        self.nl = [len(l) for l in self.ll]
    
    def PerfPCA(self, setname:str, transpose:bool=False) -> None:
        '''
        Function to peform PCA () on data and store results in self, PCA is
        performed as such that samples are along axis 0 and features along
        axis 1,
        updates self with new attributes as follows:
        - self.{setname}c{i} for i = 1, ..., n_components coordinates in
        PC-space
        - self.{setname}PC{i} for i = 1, ..., n_components Principal components
        '''
        CalcNewData = StrToData(strdata=setname)
        dataset = CalcNewData(data=self)

        trafo = self.fit_transform(
            dataset.T if transpose else dataset
        )
        for i in range(self.n_components):
            setattr(self, f'{setname}c{i+1}', trafo[:,i])
            setattr(self, f'{setname}PC{i+1}', self.components_[i,:])
        
    def PerfRecon(self, setname:str) -> None:
        '''
        Function to calculate reconstructed date and related quantities from
        PCA
        updates self with new attributes as follows:
        - self.{setname}recondata:      reconstructed data
        - self.{setname}re:             relative reconstruction error in
                                        q-space
        - self.{setname}mre:            relative mean reconstruction error
        - self.{setname}revar:          variance of relative reconstruction
                                        error in q-space
        - self.{setname}mrevar:         variance of relative mean
                                        reconstruction error
        '''
        # get data and mean of data
        CalcNewData = StrToData(strdata=setname)
        data = CalcNewData(data=self)
        mdata = np.mean(data, axis=0)

        # calculate reconstructed data
        PCspace = np.stack(
            [
                getattr(self, f'{setname}c{i+1}')
                for i in range(self.n_components)
            ], axis=1
        )
        PC = np.stack(
            [
                getattr(self, f'{setname}PC{i+1}')
                for i in range(self.n_components)
            ], axis=0
        )
        recon = np.matmul(PCspace, PC) + mdata

        # calculate difference between original and reconstructed data
        diff = data - recon

        # calculate relative reconstruction error in q-space
        re = np.sqrt(np.mean((diff/data)**2, axis=0))
        # variance of error
        revar = np.sqrt(np.mean((diff/data-re)**2, axis=0))

        # calculate relative mean reonstruction error
        mre = np.mean(re)
        # variance of error
        mrevar = np.mean(revar)

        setattr(self, f'{setname}recondata', recon)
        setattr(self, f'{setname}re', re)
        setattr(self, f'{setname}mre', mre)
        setattr(self, f'{setname}revar', revar)
        setattr(self, f'{setname}mrevar', mrevar)


        self.reconS = recon         # set reconstructed data
        self.re = re                # set relative reconstruction error
        self.mre = mre              # set relative mean reconstruction error
        self.revar = revar          # set variance of relative error
        self.mrevar = mrevar        # set variance of relative mean error
    
    def PerfFit(self, FitFunc, xdata, ydata, fitname) -> None:
        '''
        Funktion to calculate quantities from fit to data
        updates self with new attributes as follows:
        - self.{fitname}{i} fitted values
        - self.{fitname}err{i} for i = 0, ..., n-1  fit parameters
        uncertainty of fitted values
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


class StrToData():
    '''
    Class to convert string input to equivalently calculated data
    '''
    def __init__(self, strdata:str):
        self.strdata = strdata
    
    def __call__(self, data:FileData) -> np.ndarray:
        if len(self.strdata) == 0:
            raise ValueError(
                'Input string for data is empty. Please set it to a valid '
                'attribute name of FileData object or mathmatical expression.'
            )
        elif hasattr(data, self.strdata):
            if not isinstance(getattr(data, self.strdata), np.ndarray):
                raise TypeError(
                    f'Attribute "{self.strdata}" is not of type np.ndarray. '
                    'Please set it to a numpy array.'
                )
            return getattr(data, self.strdata)
        else:
            newdata = 1
            for i in range(len(self.strdata)):
                if not hasattr(data, self.strdata[i]):
                    raise AttributeError(
                        f'Data object {data} has no attribute '
                        f'{self.strdata[i]}. Note that of the current version '
                        'only multiplication between single letter attributed '
                        'datasets expressed as letter sequence is supported.'
                    )
                newdata *= getattr(data, self.strdata[i])
            return newdata


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

        if (
            not set(self.rule.Nrule).issubset([obj.N for obj in dataobjs]) or
            self.rule.Nrule == 'all'
        ):
            raise ValueError(
                f'Rule "N = {self.rule.Nrule}" not subset of all N values. '
                 'Please set Nrule to one or a list of the following values:'
                f'\n{np.unique([obj.N for obj in dataobjs])}'
                 '\nor to "all"'
            )
        
        if (
            not set(self.rule.frule).issubset([obj.f for obj in dataobjs]) or
            self.rule.frule == 'all'
        ):
            raise ValueError(
                f'Rule "f = {[f'1/{int(1/f)}' for f in self.rule.frule]}" not '
                'subset of all f values. Please set frule to one or a list of '
                'one of the following float values:'
                f'\n{np.unique([f'1/{int(1/obj.f)}' for obj in dataobjs])}'
                 '\nor to "all"'
            )
        
        if not (
            self.rule.xdata in dir(dataobjs[0])
        ):
            raise AttributeError(
                f'Rule "xdata = {self.rule.xdata}" not as attribute in '
                 'FileData objects. Please set xdata to one of the following '
                 'values:'
                f'\n{dir(dataobjs[0])}'
                '\n, specificly to "c1"/"c2" for coordinate 1/2 in PC-space '
                'or to "PC1"/"PC2" for principal component 1/2'
            )
        
        if not (
            self.rule.ydata in dir(dataobjs[0])
        ):
            raise AttributeError(
                f'Rule "ydata = {self.rule.ydata}" not as attribute in '
                 'FileData objects. Please set ydata to one of the following '
                 'values:'
                f'\n{dir(dataobjs[0])}'
                '\n, specificly to "c1"/"c2" for coordinate 1/2 in PC-space '
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
        labels = []

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
            labels.append([
                f'${rule.label}={round(getattr(obj, rule.label),2)}$'
                for obj in self.data if getattr(obj, rule.sortby) == i
            ])

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
                        label=labels[i]
                    )
                else:
                    # plot data in normal plot
                    ax.plot(
                        xdata[i], rule.scalfac*ydata[i],
                        label=labels[i]
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
                    color=rule.color[i], label=labels
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
        
        DataObj.EmpEva()                # perform empirical evaluation
        DataObj.LLCalc()                # calculate loop length and mean
        DataObj.PerfPCA(setname='S')    # perform PCA on form factor
        DataObj.PerfRecon()             # perform reconstruction of form factor

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