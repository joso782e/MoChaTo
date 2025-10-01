import numpy as np
from numpy.typing import NDArray
import scipy.linalg as lin
import scipy.optimize as opt
from sklearn.decomposition import PCA


def MeanVariance(
    obj, setname:str, axis:int, normalizedvar:bool=False
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
    data = getattr(obj, setname)
    mean = np.mean(data, axis=axis)

    if normalizedvar:
        var = np.sqrt(
            np.mean((data - mean)**2/mean**2, axis=axis)
        )
    else:
        var = np.sqrt(
            np.mean((data - mean)**2, axis=axis)
        )
    
    setattr(obj, f'mean{setname}', mean)
    setattr(obj, f'var{setname}', var)
    return mean, var

def ManipulateData(
        obj,  args:list[str], setname:str, operant:str, axis:int=None
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
        dataset = getattr(obj, args[0])
    else:
        for arg in args:
            argset = getattr(obj, arg)
            if concataxis:
                dataset = [dataset, argset]
                dataset = operant(dataset, axis=concataxis)
            else:
                dataset = operant(dataset, argset)
        setattr(obj, setname, dataset)

def ExtremaInflection(obj, xdata:str, ydata:str, order:int):
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
    if not hasattr(obj, xdata):
        raise AttributeError(
            f'FileData object  does not have {xdata} as attribute. Please '
            'give a valid attribute name for "xdata".'
        )
    if not hasattr(obj, ydata):
        raise AttributeError(
            f'FileData object  does not have {ydata} as attribute. Please '
            'give a valid attribute name for "ydata".'
        )
    xvalues = getattr(obj, xdata)
    yvalues = getattr(obj, ydata)

    extrema = [
        Extrema(xvalues, yvalues[i,:] , order)
        for i in range(yvalues.shape[0])
    ]

def SolveEigenProblem(obj, matrices:str) -> tuple[list]:
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
    if not(hasattr(obj, matrices)):
        raise AttributeError(
            f'Object does not have attribute {matrices}. Please choose a '
            'string for matrices according to the methods doc string.'
        )
    eigv = []
    eigf = []
    specrad = []
    
    for m in getattr(obj, matrices):
        eigv1, eigf1 = lin.eigh(m)

        eigv.append(eigv1[np.abs(eigv1) > 1e-10])
        eigf.append(eigf1[np.abs(eigv1) > 1e-10])
        specrad.append(np.max(np.abs(eigv1)))
                
    setattr(obj, f'{matrices}eigv', eigv)
    setattr(obj, f'{matrices}eigf', eigf)
    setattr(obj, f'{matrices}specrad', specrad)

    return eigv, eigf, specrad

def Trace(obj, matrices:str) -> list[float]:
    '''
    
    '''
    if hasattr(obj, f'{matrices}eigv'):
        eigv = getattr(obj, f'{matrices}eigv')
    else:
        eigv, _ = SolveEigenProblem(obj, matrices=matrices)
    
    trace = [np.sum(m) for m in eigv]

    setattr(obj, f'{matrices}trace', trace)

    return trace

def Determinant(obj, matrices:str) -> list[float]:
    '''
    Function to compute pseudo-determinant of a list of matrices using
    their eigenvalues
    '''
    if hasattr(obj, f'{matrices}eigv'):
        eigv = getattr(obj, f'{matrices}eigv')
    else:
        eigv, _ = SolveEigenProblem(obj, matrices=matrices)

    det = [np.prod([m for m in eig if m != 0]) for eig in eigv]

    setattr(obj, f'{matrices}det', det)

    return det


def ScaleData(obj, setname:str, scalfac:NDArray[np.float64]):
    '''
    
    '''

    data = getattr(obj, setname)
    setattr(obj, f'scaled{setname}', scalfac*data)
    return scalfac*data

def PerfPCA(
        obj, setname:str, transpose:bool=False
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
    dataset = getattr(obj, setname)
    pcaobj = obj.pcaobj
    trafo = pcaobj.fit_transform(
        dataset.T if transpose else dataset
    )
    for i in range(pcaobj.n_components):
        setattr(obj, f'{setname}c{i+1}', trafo[:,i])
        setattr(obj, f'{setname}PC{i+1}', pcaobj.components_[i,:])


def PerfRecon(obj, setname:str, normalize:bool=True) -> None:
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
    ncomps = obj.pcaobj.n_components

    # get data and mean of data
    data = getattr(obj, setname)
    mdata = np.mean(data, axis=0)

    # calculate reconstructed data
    PCspace = np.stack(
        [
            getattr(obj, f'{setname}c{i+1}')
            for i in range(ncomps)
        ], axis=1
    )
    PC = np.stack(
        [
            getattr(obj, f'{setname}PC{i+1}')
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

    setattr(obj, f'recondata{setname}', recon)
    setattr(obj, f're{setname}', re)
    setattr(obj, f'mre{setname}', mre)
    setattr(obj, f'revar{setname}', revar)
    setattr(obj, f'mrevar{setname}', mrevar)


def PerfFit(
        obj, FitFunc, xdata:str, ydata:str, fitname:str,
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
    if not hasattr(obj, xdata):
        raise AttributeError(
            f'Input "{xdata}" for xdata not as attribute in FileData '
            'object. Please set xdata to one of the following values:'
            f'\n{dir(obj)}'
            '\n, specificly to "c1"/"c2" for coordinate 1/2 in PC-space '
            'or to "PC1"/"PC2" for principal component 1/2'
        )
    if not hasattr(obj, ydata):
        raise AttributeError(
            f'Input "{ydata}" for ydata not as attribute in FileData '
            'object. Please set ydata to one of the following values:'
            f'\n{dir(obj)}'
            '\n, specificly to "c1"/"c2" for coordinate 1/2 in PC-space '
            'or to "PC1"/"PC2" for principal component 1/2'
        )
    
    xlower = -np.inf if xlim[0] is None else xlim[0]
    xupper = np.inf if xlim[1] is None else xlim[1]

    # get xdata and ydata from self
    xdata = getattr(obj, xdata)
    ydata = getattr(obj, ydata)

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
                
                setattr(obj, f'{fitname}{j}', [fit[j]])
        elif i == ydata.shape[0] - 1:
            for j in range(len(fit)):
                f = getattr(obj, f'{fitname}{j}')

                f.append(fit[j])

                setattr(obj, f'{fitname}{j}', np.array(f))
        else:
            for j in range(len(fit)):
                f = getattr(obj, f'{fitname}{j}')

                f.append(fit[j])

                setattr(obj, f'{fitname}{j}', f)