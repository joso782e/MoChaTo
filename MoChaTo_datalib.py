'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
MoChaTo: Library for data evaluation of SCNPs
-------------------------------------------------------------------------------
written by Jonas Soucek, 2025
at TU Dresden and Leibniz Institute of Polymer Research
'''

# import necessary modules
import json
import h5py
import numpy as np
from numpy.typing import NDArray
import scipy.optimize as opt
import scipy.linalg as lin
import scipy.signal as sig
from sklearn.decomposition import PCA


# print import statement if the module has been imported successfully
print("Module \"MoChaTo_datalib.py\" imported successfully")


with open('.\\data_evaluation\\Scripts\\config.json', 'r') as configf:
    config = json.load(configf)


class FileData(PCA):
    '''
    Class to store and update results for comprehensive file evaluation after 
    individual file evaluation. It inherits from the PCA class of sklearn and
    contains the following attributes after initialization:

    - condi:        string describing simulated condition
    - N:            chainlength in number of monomers
    - position:     monomer positions
    - f:            connector distance
    - q:            values in q-space
    - S:            form factor in q-space
    - clmat:        crosslink matrix
    - sequence:     monomer sequence

    calling further class methods may extend those attributes with new ones
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
        sequence:np.ndarray,
        n_components:int
    ):
        self.pcaobj = PCA(n_components=n_components)
        self.condi = condi
        self.N = N
        self.positions = np.linspace(1, N, N)    # monomer positions in chain
        self.f = 1/f
        self.q = q
        self.S = S
        self.clmat = np.asanyarray(clmat, dtype=np.int_)
        self.sequence = sequence

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

        # get monomer sequence
        sequence = np.squeeze(file[path[:-l]+'sequence'])

        return FileData(
            n_components=ncomps, condi=Cond, N=NChain, f=InterconDens, q=qKey,
            S=SData, clmat=crosslinks, sequence=sequence
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
        self.mll = np.asarray([np.mean(l) if len(l) > 0 else 0 for l in ll])
        # set number of loops
        self.nl = np.asarray([len(l) for l in self.ll])

    def LoopBalanceProfile(self) -> None:
        '''
        Function to compute a balance profile, balance and standard deviation
        for a given sequence

        Input:
        
        Output:
        updates self with new attributes as follows:
        - self.loopprofiles...          balance profile over sequences
        - self.loopbalances...          normalized and centered balances of
                                        profiles
        - self.loopdevbalances...       normalized deviation of balances
        '''
        sequence = self.positions
        clmat = self.clmat
        
        profiles = []
        balances = []
        devbalances = []

        for i in range(clmat.shape[0]):
            cls = [
                (clmat[i,k,1], clmat[i,k,2]) for k in range(clmat.shape[1])
                if clmat[i,k,0]
            ]

            p, bp, devb = BalanceProfile(
                sequence=sequence, cls=cls
            )

            profiles.append(p)
            balances.append(bp)
            devbalances.append(devb)
        
        profiles = np.stack(profiles, axis=0)
        balances = np.stack(balances, axis=0)
        devbalances = np.stack(devbalances, axis=0)

        setattr(self, f'loopprofiles', profiles)
        setattr(self, f'loopbalances', balances)
        setattr(self, f'loopdev', devbalances)

    def Blockiness(self, blocktype:int, normalize:bool) -> None:
        '''
        Function to calculate blockiness of SCNP from sequences

        Input:
        blocktype (dtype = int)...          type of block interface,
                                        1 for blockiness of crosslinker
                                        sequence 
                                        2 for reactive blockiness
                                        3 for crosslinker/backbone blockiness
                                        4 for activeted blockiness
                                        5 for total blockiness
        normalize (dtype = bool)...     wether blockiness should be normalized
                                        (True) or not (False)
        
        Output:
        updates self with new attributes as follows:
        - self.b{blocktype}...          blockiness type x of SCNPs
        '''
        sequence = self.sequence
        b = np.apply_along_axis(
            lambda x: Blockiness(
                sequence=x, blocktype=blocktype, f=self.f, normalize=normalize
            ), axis=1, arr=sequence
        )

        setattr(self, f'b{blocktype}', b)

    def RouseMatrix(self) -> NDArray:
        '''
        Function to compute the generalized Rouse matrix for all polymers

        Input:
        

        Output:
        updates self with new attributes as follows:
        - self.rouse...             generalized rouse matrix of SCNPs
        '''
        clmat = self.clmat
        rouse = []
        
        for i in range(clmat.shape[0]):
            cls = [
                (clmat[i,j,1], clmat[i,j,2]) for j in range(clmat.shape[1])
                if clmat[i,j,0] != 0
            ]
            rouse.append(RouseMatrix(self.N, cls))
        
        self.rouse = rouse
        return rouse
    
    def SwellingRatio(self) -> NDArray:
        '''
        Function to compute swelling ratio or chain ideality

        Input:

        Output:
        updates self with new attributes as follows:
        - self.swellingratio...         swelling ratio/ideality degree of SCNPs
        '''
        if not hasattr(self, 'Rg1'):
            def FitGyraRad(x, a0, a1):
                '''
                Function for fitting radii of gyration to form factor 
                '''
                return a0 - 1/3*(a1*x)**2
            
            self.PerfFit(
                FitFunc=FitGyraRad, xdata='q', ydata='S', fitname='Rg',
                xlim=(None, 2e-2)
            )

        if not hasattr(self, 'rouse'):
            self.RouseMatrix()

        if not hasattr(self, 'rouseeigv'):
            self.SolveEigenProblem(matrices='rouse')

        l = 2.734*5.19
        
        Q = [
            (self.Rg1[i]*self.N**0.5/l/(np.sum(1/self.rouseeigv[i]))**0.5)**3
            for i in range(len(self.rouseeigv))
        ]

        setattr(self, 'swellingratio', Q)
        return Q
    
    def ConClassRatio(self) -> tuple[list,list,list]:
        '''
        Function to compute the conformation class ratios by the type 1 loop
        conformation convention for several molecules
        conformation class:
        - 1: loops in series
        - 2: loops entangled
        - 3: loops parallel

        Input:

        Output:
        updates self with new attributes as follows:
        - self.conratio1...         ratio for loops in series
        - self.conratio2...         ratio for entagled loop conformations
        - self.conratio3...         ratio for parallel loop conformations
        '''
        clmat = self.clmat

        conratio1 = []
        conratio2 = []
        conratio3 = []


        for k in range(clmat.shape[0]):
            cls = [
                (clmat[k,n,1], clmat[k,n,2]) for n in range(clmat.shape[1])
                if clmat[k,n,0]
            ]

            p1, p2, p3 = ConClassRatio(cls)

            conratio1.append(p1)
            conratio2.append(p2)
            conratio3.append(p3)

        self.conratio1 = conratio1
        self.conratio2 = conratio2
        self.conratio3 = conratio3

        return conratio1, conratio2, conratio3

    def MeanVariance(
        self, setname:str, axis:int, normalizedvar:bool=False
    ) -> tuple[NDArray]:
        '''
        Function to compute mean and variance of an arbitrary dataset along a
        given axis

        Input:
        setname (dtype = str)...
        axis (dtype = int)...
        normalizedvar (dtype = bool)...

        Output:

        
        updates self with new attributes as follows:
        - self.mean{setname}:          mean of dataset
        - self.var{setname}:        variannce of dataset
        '''
        data = getattr(self, setname)
        mean = np.mean(data, axis=axis)

        if normalizedvar:
            var = np.sqrt(
                np.mean((data - mean)**2/mean**2, axis=axis)
            )
        else:
            var = np.sqrt(
                np.mean((data - mean)**2, axis=axis)
            )
        
        setattr(self, f'mean{setname}', mean)
        setattr(self, f'var{setname}', var)
        return mean, var

    def ManipulateData(
            self,  args:list[str], setname:str, operant:str, axis:int=None
        ):
        '''
        Function to manipulate excisting datasets

        Input
        - args (dtype = list[str])...   list of attributes to be used to
                                        manipulate data for PCA
        - setname (dtype = str)...      category attribute name for results,
                                        see description of output
        - operant (dtype = str)...      operant to be used to manipulate data
                                        before PCA, can be '+', '-', '*',
                                        'concatenate' or 'None' for no
                                        manipulation
        - axis (dtype = int)...         if operant is 'concatenate'
                                        specifies on which axis to concatenate

        Output:
        updates self with new attributes as follows:
        '''
        concataxis = None
        if operant == '+':
            operant = np.add
            dataset = 0
        elif operant == '-':
            operant = np.subtract
            dataset = 0
        elif operant == '*':
            operant = np.multiply
            dataset = 1
        elif operant == 'concatenate':
            operant = np.concat
            concataxis = axis
        
        if operant not in [np.add, np.subtract, np.multiply]:
            dataset = getattr(self, args[0])
        else:
            for arg in args:
                argset = getattr(self, arg)
                if concataxis:
                    dataset = [dataset, argset]
                    dataset = operant(dataset, axis=concataxis)
                else:
                    dataset = operant(dataset, argset)
            setattr(self, setname, dataset)

    def ExtremaInflection(self, xdata:str, ydata:str, order:int):
        '''
        Function to calculate extrema and points of inflection

        Input:
        xdata (dtype = str)...          str describing atribute of FileData
                                        object containing x-values
        ydata (dtype = str)...          str describing atribute of FileData
                                        object containing y-values
                                
        Output:
        updates self with new attributes as follows:
        - self.ext{ydata}...            ndarray containing all extrema j with
                                        x-coordinate in [i,j] and y-coordinate
                                        in [i,j+1]
        - self.nmax{ydata}...           number of maxima in ydata
        - self.nmin{ydata}...           number of minima in ydata
        - self.inf{ydata}...            ndarray containing all inflections j
                                        with x-coordinate in [i,j] and
                                        y-coordinate in [i,j+1]
        '''
        if not hasattr(self, xdata):
            raise AttributeError(
               f'FileData object  does not have {xdata} as attribute. Please '
                'give a valid attribute name for "xdata".'
            )
        if not hasattr(self, ydata):
            raise AttributeError(
               f'FileData object  does not have {ydata} as attribute. Please '
                'give a valid attribute name for "ydata".'
            )
        xvalues = getattr(self, xdata)
        yvalues = getattr(self, ydata)

        extrema = [
            Extrema(xvalues, yvalues[i,:] , order)
            for i in range(yvalues.shape[0])
        ]

    def SolveEigenProblem(self, matrices:str) -> tuple[list]:
        '''
        Function to solve eigenvalue problem for a list of matrices using
        eigh from scipy.linalg and store result in self

        Input:
        matrices (dtype = str)...           str of attribute which should
                                            contain a list with several
                                            matrices for which to compute
                                            eigenvalues and eigenvectors
        
        Output:
        eigv (dtype = list[NDArray])...     list containing arrays of
                                            eigenvalues for each matrix
        eigf (dtype = list[NDArray])...     list containing arrays of
                                            eigenvectors for each matrix
        
        updates self with new attributes as follows:
        - self.{matrices}eigv...
        - self.{matrices}eigf...
        - self.{matrices}specrad...
        '''
        if not(hasattr(self, matrices)):
            raise AttributeError(
               f'Object does not have attribute {matrices}. Please choose a '
                'string for matrices according to the methods doc string.'
            )
        eigv = []
        eigf = []
        specrad = []
        
        for m in getattr(self, matrices):
            eigv1, eigf1 = lin.eigh(m)

            eigv.append(eigv1[np.abs(eigv1) > 1e-10])
            eigf.append(eigf1[np.abs(eigv1) > 1e-10])
            specrad.append(np.max(np.abs(eigv1)))
                    
        setattr(self, f'{matrices}eigv', eigv)
        setattr(self, f'{matrices}eigf', eigf)
        setattr(self, f'{matrices}specrad', specrad)

        return eigv, eigf
    
    def Trace(self, matrices:str) -> list[float]:
        '''
        
        '''
        if hasattr(self, f'{matrices}eigv'):
            eigv = getattr(self, f'{matrices}eigv')
        else:
            eigv, _ = self.SolveEigenProblem(matrices=matrices)
        
        trace = [np.sum(m) for m in eigv]

        setattr(self, f'{matrices}trace', trace)

        return trace
    
    def Determinant(self, matrices:str) -> list[float]:
        '''
        Function to compute pseudo-determinant of a list of matrices using
        their eigenvalues
        '''
        if hasattr(self, f'{matrices}eigv'):
            eigv = getattr(self, f'{matrices}eigv')
        else:
            eigv, _ = self.SolveEigenProblem(matrices=matrices)

        det = [np.prod([m for m in eig if m != 0]) for eig in eigv]

        setattr(self, f'{matrices}det', det)

        return det

    def BinData(
            self, xdata:str, ydata:str, bins:np.ndarray=20, ppb:bool=False,
            xlower:float=None, xupper:float=None
    ) -> None:
        '''
        Function to bin ydata along given xdata axis, calculate their mean and
        variance for each bin seperately

        Input:
        xdata (dtype = str)...          str describing atribute of FileData
                                        object containing x-values
        ydata (dtype = str)...          str describing atribute of FileData
                                        object containing y-values
        bins (dtype = np.ndarray)...    int or array-like
                                        - if int: sets number of evenly spaced
                                        bins
                                        - if array-like: sets bin edges
        ppb (dtype = bool)...           if True, computes bins so that equal
                                        number of data points per bin
                                        using bins as integer

        Output:
        updates self with new attributes as follows:
        - self.binmean{xdata}...
        - self.binmean{ydata}...
        - self.binerr{xdata}...
        - self.binerr{ydata}...
        - self.binmeanerr{xdata}...
        - self.binmeanerr{ydata}...
        '''
        xvalues = np.asarray(getattr(self, xdata))
        yvalues = np.asarray(getattr(self, ydata))
        
        diff = np.max(xvalues) - np.min(xvalues)

        xmin = xlower if xlower else np.min(xvalues) - 0.001*diff
        xmax = xupper if xupper else np.max(xvalues) + 0.001*diff

        if ppb:
            if not isinstance(bins, int):
                raise TypeError(
                    'If ppb is True, bins must be an integer specifying the '
                    'number of data points per bin.'
                )
            sort = np.argsort(xvalues)
            sort = xvalues[sort]
            bins = sort[::bins]
        else:
            if isinstance(bins, int):
                bins = np.linspace(xmin, xmax, bins+1)

        bins[0] = xmin
        bins[-1] = xmax

        y = np.zeros_like(bins[1:])
        yerr = np.zeros_like(bins[1:])
        xerr = np.zeros_like(bins[1:])
        xmeanerr = np.zeros_like(bins[1:])
        ymeanerr = np.zeros_like(bins[1:])

        for i in range(len(bins[1:])):
            binned = yvalues[xvalues >= bins[i]]
            buffer = xvalues[xvalues >= bins[i]]
            binned = binned[buffer <= bins[i+1]]
            buffer = buffer[buffer <= bins[i+1]]
            if len(binned):
                y[i] = np.mean(binned)
                if len(binned) >= 10:
                    yerr[i] = np.sqrt(np.var(binned, ddof=1))
                    xerr[i] = np.sqrt(np.var(buffer, ddof=1))
                else:
                    yerr[i] = (max(binned) - min(binned))/2
                    xerr[i] = (max(buffer) - min(buffer))/2
                ymeanerr[i] = yerr[i]/np.sqrt(len(binned))
                xmeanerr[i] = xerr[i]/np.sqrt(len(binned))
            
        idx0 = np.nonzero(y)
        bins = (bins[1:] + bins[:-1])/2
        
        
        setattr(self, f'binmean{xdata}', bins[idx0])
        setattr(self, f'binmean{ydata}', y[idx0])
        setattr(self, f'binerr{xdata}', xerr[idx0])
        setattr(self, f'binerr{ydata}', yerr[idx0])
        setattr(self, f'binmeanerr{xdata}', xmeanerr[idx0])
        setattr(self, f'binmeanerr{ydata}', ymeanerr[idx0])

    def ScaleData(self, setname:str, scalfac:NDArray[np.float64]):
        '''
        
        '''

        data = getattr(self, setname)
        setattr(self, f'scaled{setname}', scalfac*data)
        return scalfac*data

    def PerfPCA(
            self, setname:str, transpose:bool=False
        ) -> None:
        '''
        Function to peform PCA () on data and store results in self, PCA is
        performed as such that samples are along axis 0 and features along
        axis 1

        Input:
        - setname (dtype = str)...      attribute name of dataset to perform
                                        PCA on
        - transpose (dtype = bool)...   if True, data is transposed before
                                        PCA, default is False

        Output:
        updates self with new attributes as follows:
        - self.{setname}c{i} for i = 1, ..., n_components coordinates in
        PC-space
        - self.{setname}PC{i} for i = 1, ..., n_components Principal components
        '''
        dataset = getattr(self, setname)
        pcaobj = self.pcaobj
        trafo = pcaobj.fit_transform(
            dataset.T if transpose else dataset
        )
        for i in range(pcaobj.n_components):
            setattr(self, f'{setname}c{i+1}', trafo[:,i])
            setattr(self, f'{setname}PC{i+1}', pcaobj.components_[i,:])        
        
    def PerfRecon(self, setname:str, normalize:bool=True) -> None:
        '''
        Function to calculate reconstructed date and related quantities from
        PCA

        Input:
        setname (dtype = str)...        category attribute name of dataset to
                                        perform reconstruction from PCA
        normalize 

        Output:
        updates self with new attributes as follows:
        - self.recondata{setname}:      reconstructed data
        - self.re{setname}:             relative reconstruction error in
                                        q-space
        - self.mre{setname}:            relative mean reconstruction error
        - self.revar{setname}:          variance of relative reconstruction
                                        error in q-space
        - self.mrevar{setname}:         variance of relative mean
                                        reconstruction error
        '''
        # get number of components from PCA object
        ncomps = self.pcaobj.n_components

        # get data and mean of data
        data = getattr(self, setname)
        mdata = np.mean(data, axis=0)

        # calculate reconstructed data
        PCspace = np.stack(
            [
                getattr(self, f'{setname}c{i+1}')
                for i in range(ncomps)
            ], axis=1
        )
        PC = np.stack(
            [
                getattr(self, f'{setname}PC{i+1}')
                for i in range(ncomps)
            ], axis=0
        )
        recon = np.matmul(PCspace, PC) + mdata

        # calculate difference between original and reconstructed data
        diff = data - recon

        # calculate reconstruction error in q-space and normalize if specified
        if normalize:
            re = np.sqrt(np.mean((diff/data)**2, axis=0))
            # variance of error
            revar = np.sqrt(np.mean((diff/data-re)**2, axis=0))
        else:
            re = np.sqrt(np.mean((diff)**2, axis=0))
            # variance of error
            revar = np.sqrt(np.mean((diff-re)**2, axis=0))

        # calculate relative mean reonstruction error
        mre = np.mean(re)
        # variance of error
        mrevar = np.mean(revar)

        setattr(self, f'recondata{setname}', recon)
        setattr(self, f're{setname}', re)
        setattr(self, f'mre{setname}', mre)
        setattr(self, f'revar{setname}', revar)
        setattr(self, f'mrevar{setname}', mrevar)
    
    def PerfFit(
            self, FitFunc, xdata:str, ydata:str, fitname:str,
            xlim:tuple[float]=(None, None)
    ) -> None:
        '''
        Funktion to calculate quantities from fit to data

        Input:
        FitFunc (callable)...       callable representing the function to be
                                    fitted, for further reference see
                                    documentation of scipy.optimize.curve_fit()
        xdata (dtype = str)...      str describing atribute of FileData object
                                    containing x-values
        ydata (dtype = str)...      str describing atribute of FileData object
                                    containing x-values
        fitname (dtype = str)...    string determining attribute name in
                                    FileData object for fit parameters

        Output:
        updates self with new attributes as follows:
        - self.{fitname}{i} fitted values
        - self.{fitname}err{i} for i = 0, ..., n-1  fit parameters
        uncertainty of fitted values
        '''
        # check if FitFunc is callable
        if not callable(FitFunc):
            raise TypeError(
                f'Input "{FitFunc}" for FitFunc not callable. '
                'Please set FitFunc to a callable function'
            )
        # check if xdata and ydata are attributes of self
        if not hasattr(self, xdata):
            raise AttributeError(
                f'Input "{xdata}" for xdata not as attribute in FileData '
                'object. Please set xdata to one of the following values:'
                f'\n{dir(self)}'
                '\n, specificly to "c1"/"c2" for coordinate 1/2 in PC-space '
                'or to "PC1"/"PC2" for principal component 1/2'
            )
        if not hasattr(self, ydata):
            raise AttributeError(
                f'Input "{ydata}" for ydata not as attribute in FileData '
                'object. Please set ydata to one of the following values:'
                f'\n{dir(self)}'
                '\n, specificly to "c1"/"c2" for coordinate 1/2 in PC-space '
                'or to "PC1"/"PC2" for principal component 1/2'
            )
        
        xlower = -np.inf if xlim[0] is None else xlim[0]
        xupper = np.inf if xlim[1] is None else xlim[1]

        # get xdata and ydata from self
        xdata = getattr(self, xdata)
        ydata = getattr(self, ydata)

        ydata = ydata[:, xdata >= xlower]
        xdata = xdata[xdata >= xlower]

        ydata = ydata[:, xdata <= xupper]
        xdata = xdata[xdata <= xupper]
                
        for i in range(ydata.shape[0]):
            fit, _ = opt.curve_fit(
                FitFunc, xdata=xdata, ydata=ydata[i,:]
            )

            if i == 0:
                for j in range(len(fit)):
                    
                    setattr(self, f'{fitname}{j}', [fit[j]])
            elif i == ydata.shape[0] - 1:
                for j in range(len(fit)):
                    f = getattr(self, f'{fitname}{j}')

                    f.append(fit[j])

                    setattr(self, f'{fitname}{j}', np.array(f))
            else:
                for j in range(len(fit)):
                    f = getattr(self, f'{fitname}{j}')

                    f.append(fit[j])

                    setattr(self, f'{fitname}{j}', f)

        
def filter_func(
        name:str, file:h5py.File, NComps:int=config['NComps'],
        filter_obj:str=config['filter_obj']
) -> FileData:
    '''
    Function to filter data groups in .hdf5 'input_file' that contain the
    string 'filter_obj' and perform data collection, evaluation and plotting
    on them

    Input:
    name (dtype = str)...           name of data group to be filtered
    file (dtype = h5py.File)...     input file
    NComps (dtype = int)...         number of principal components to perform 
                                    PCA with
    filter_obj (dtype = str)...     name of one data group to filter simulated
                                    conditions in .hdf5 'input_file'    

    Output:
    DataObj or None
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

        # print separator line to indicate end of one condition evaluation in 
        # terminal for debugging
        print('-'*79)            
        return DataObj
    
    else:
        return None


def Blockiness(
        sequence:list[int], blocktype:int, f:float, normalize:bool
    ) -> float:
    '''
    Function to calculate blockiness of sequence

    Input:
    sequence (dtype = list[int])...     list of integers representing sequence
    blocktype (dtype = int)...          type of block interface,
                                        1 for blockiness of crosslinker
                                        sequence 
                                        2 for reactive blockiness
                                        3 for crosslinker blockiness
                                        4 for activeted blockiness
                                        5 for total blockiness
    normalize (dtype = bool)...         wether blockiness should be normalized
                                        (True) or nor (False)

    Output:
    blockiness (dtype = float)...       blockiness of sequence,
                                        normalized to sequence length
    '''
    # create boolean sequence depending on blocktype
    if blocktype > 4:
        f = 2*f
        bmin = np.abs(2*(f - 0.5))
        v = sequence[:-1] == sequence[1:]
    elif blocktype > 1:
        if blocktype == 3:
            f = 2*f
        sequence = sequence == blocktype
        bmin = np.abs(2*(f - 0.5))
        v = sequence[:-1] == sequence[1:]
    else:
        sequence = np.array([s for s in sequence if s != 3])
        sequence = sequence == 2
        bmin = 0
        v = sequence[:-1] == sequence[1:]


    # apply binding tensor on boolean sequence
    b = (1 + np.sum(v))/len(sequence)

    if normalize:
        if (b - 1):
            # normalize, if b is unequal to 1
            b = (b - bmin)/(1 - bmin)
    
    return b


def BalanceProfile(
        sequence:np.ndarray, cls:list[tuple]
    ) -> np.ndarray:
    '''
    Function to compute the balance profile over a sequence for a given
    criterion

    Input:
    sequence (dtype = np.ndarray)...    sequence over which to compute the
                                        balance profile
    cls (dtype = list)...               list of tuples containing crosslink
                                        indices

    Output:
    '''
    m = len(cls)
    n = len(sequence)

    lower = [cl[0] if cl[0] < cl[1] else cl[1] for cl in cls]
    upper = [cl[1] if cl[1] > cl[0] else cl[0] for cl in cls]
    
    profile = np.full(shape=(m,n), fill_value=sequence, dtype=float)
    lower = np.stack([lower for _ in range(n)], axis=1)
    upper = np.stack([upper for _ in range(n)], axis=1)

    # compute balance profile
    profile = (profile >= lower) == (profile <= upper)
    profile = np.sum(profile, axis=0)

    # calculate balance point of profile and standard deviation
    if np.sum(profile) == 0:
        b = 1.5
        devb = 0
    else:
        b = np.sum(np.multiply(profile, sequence))/np.sum(profile)
        devb = np.sqrt(
            np.sum((np.multiply(profile, sequence)/np.sum(profile) - b)**2)
        )/n

        b = np.abs(2*b/n - 1)           # normalized and centered balance on
                                        # sequence
    
    return np.squeeze(profile), float(b), float(devb/n)


def Extrema(
    xdata:np.ndarray, ydata:np.ndarray, order:int
) -> np.ndarray:
    '''
    Function to compute the extrema of a series

    Input:
    xdata (dtype = np.ndarray)...   array with x-values
    ydata (dtype = np.ndarray)...   array with y-values

    Output:
    extrema...                      ndarray containing all extrema i with 
                                    x-coordinate in [i] and y-coordinate
                                    in [i+1]
    '''
    maxind = sig.argrelmax(ydata, order=order)[0]
    minind = sig.argrelmin(ydata, order=order)[0]

    xtrem = np.concat([xdata[maxind], xdata[minind]])
    ytrem = np.concat([ydata[maxind], ydata[minind]])

    ytrem = ytrem[np.argsort(xtrem)]
    xtrem = np.argsort(xtrem)

    extrema = np.array(list(zip(xtrem, ytrem)))
    print(extrema)

    return extrema.flatten()


def Inflection(
    xdata:np.ndarray, ydata:np.ndarray, order:int
) -> tuple[np.ndarray, int, int]:
    '''
    Function to compute the inflection points of a series
    
    Input:
    xdata (dtype = np.ndarray)...   array with x-values
    ydata (dtype = np.ndarray)...   array with y-values

    Output:
    inflection...                   ndarray containing all inflection points i
                                    with x-coordinate in [i] and y-coordinate
                                    in [i+1]
    '''
    ydata = np.diff(ydata)

    return Extrema(xdata[:-1], ydata, order=order)


def RouseMatrix(N:int, cls:list[tuple]=[]) -> NDArray:
    '''
    Function to compute the generalized Rouse matrix from a given crosslinker
    sequence asuming linear chain topology. The Rouse matrix is constructed by
    starting with a zero matrix. For every link between i and j the matricies
    elements are updated as
    follows:
        M_ii += 1
        M_jj += 1
        for diagonal elements and
        M_ij -= 1
        M_ji -= 1
        for non-diagonal elements
    (see Sommer and Blumen on generalized Gaussian structures)

    Input:
    N (dtype = int)...              number of monomers
    linchain (dtype = bool)...      boolean wether constructing Rouse matrix
                                    for linear chain (True) molecule or not
                                    (False)
    cls (dtype = list[tuple])...     list of tuples, each tuple contains the
                                    link's monomers

    Output:
    rouse (dtype = NDArray)...      the constructed Rouse matrix, it is of
                                    dimension N x N
    '''
    rouse = np.full(N, 2)
    rouse = np.diag(rouse) - np.diag(rouse[1:]-1, k=1) - np.diag(rouse[1:]-1, k=-1)
    rouse[0,0] = 1
    rouse[-1,-1] = 1

    if len(cls) == 0:
        # if no crosslinker sequence is given, return the Rouse matrix
        return rouse
    
    for cl in cls:
        i = int(cl[0] - 1)
        j = int(cl[1] - 1)
        rouse[i,i] += 1
        rouse[j,j] += 1
        rouse[i,j] -= 1
        rouse[j,i] -= 1
    return rouse


def FiniteSum(kterm, kstart:int, kstop:int) -> float:
    '''
    Function
    '''
    finalsum = 0
    for k in range(kstart, kstop+1):
        finalsum += kterm(k)

    return finalsum


def ConformationType1(i:tuple[int,int], j:tuple[int,int]) -> int:
    '''
    Function to compare two loops with each other and determine their
    conformation class in the type 1 convention, further explanations can be
    found in the Bachelor thesis

    Input:
    i (dtype = tuple)...        start and end values of first loop
    j (dtype = tuple)...        start end end values of second loop

    Output:
    conclass (dtype = int)...   conformation class:
                                 - 1: loops in series
                                 - 2: loops entangled
                                 - 3: loops parallel
    '''
    if i[0] < i[1]:
        istart = i[0]
        iend = i[1]
    else:
        istart = i[1]
        iend = i[0]
    
    if j[0] < j[1]:
        jstart = j[0]
        jend = j[1]
    else:
        jstart = j[1]
        jend = j[0]
    
    if jstart < istart:
        istart, jstart = jstart, istart
        iend, jend = jend, iend

    conclass = 1

    if iend > jstart:
        conclass += 1
    if iend > jend:
        conclass += 1

    return conclass  


def ConClassRatio(cls:list[tuple]) -> tuple[float,float,float]:
    '''
    Function to compute the conformation class ratios by the type 1 loop
    conformation convention, further explanations can be found in the Bachelor
    thesis
    conformation class:
        - 1: loops in series
        - 2: loops entangled
        - 3: loops parallel

    Input:
    cls (dtype = list[tuple])...    list of tuples, each tuple contains the
                                    link's monomers

    Output:
    p1 (dtype = float)...           ration of conformation class 1
    p2 (dtype = float)...           ration of conformation class 2
    p3 (dtype = float)...           ration of conformation class 3
    '''
    L = len(cls)
    C = L*(L - 1)/2
    if C == 0:
        return 0, 0, 0
    loopcon = np.zeros((L, L), dtype=int)

    for ind, _ in np.ndenumerate(loopcon):
        i = cls[ind[0]]
        j = cls[ind[1]]
        if i == j:
            continue

        loopcon[ind] = ConformationType1(i, j)
    
    p1 = np.sum(loopcon == 1)/2/C
    p2 = np.sum(loopcon == 2)/2/C
    p3 = np.sum(loopcon == 3)/2/C

    return p1, p2, p3