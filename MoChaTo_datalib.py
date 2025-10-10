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


with open('config.json', 'r') as configf:
    config = json.load(configf)


class FileData:
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
    ):
        self.condi = condi
        self.N = N
        self.positions = np.linspace(1, N, N)    # monomer positions in chain
        self.f = 2/f
        self.q = q
        self.S = S
        self.clmat = np.asanyarray(clmat, dtype=np.int_)
        self.sequence = sequence

    @staticmethod
    def ExtractFileData(
        file: h5py.File,
        path: str,
        filter_obj: str,
    ) -> 'FileData':

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
            condi=Cond, N=NChain, f=InterconDens, q=qKey,
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
        - self.loopdev...               normalized deviation of balances
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

    def TopoRatio(
        self,
        minLoopLength: int = 0,
        nextNeighbor: bool = False,
    ) -> tuple[list,list,list]:
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
        - self.toporatio1...         ratio for loops in series
        - self.toporatio2...         ratio for entagled loop conformations
        - self.toporatio3...         ratio for parallel loop conformations
        '''
        clmat = self.clmat
    
        toporatio1 = []
        toporatio2 = []
        toporatio3 = []


        for k in range(clmat.shape[0]):
            cls = [
                (clmat[k,n,1], clmat[k,n,2]) for n in range(clmat.shape[1])
                if clmat[k,n,0]
            ]
            cls = [
                (alpha, omega) if alpha < omega else (omega, alpha) for (alpha, omega) in cls
            ]

            p1, p2, p3 = TopoRatio(cls, minLoopLength, nextNeighbor)

            toporatio1.append(p1)
            toporatio2.append(p2)
            toporatio3.append(p3)

        self.toporatio1 = toporatio1
        self.toporatio2 = toporatio2
        self.toporatio3 = toporatio3

        return toporatio1, toporatio2, toporatio3

def filter_func(
        name: str,
        file: h5py.File,
        filter_obj: str = config['filter_obj']
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
                                           filter_obj=filter_obj)        

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
    f = f/2
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
        normprof = profile/np.sum(profile)
        b = np.sum(normprof)
        sig = np.sum((sequence - b)**2)/(n - 1)
        w = np.sum(normprof)/n
        w2 = np.sum(normprof**2)/n

        devb = sig*w2/w**2

        b = np.abs(2*b/n - 1)           # normalized and centered balance on
                                        # sequence
    
    return np.squeeze(profile), float(b), float(np.sqrt(devb)/n)


def RouseMatrix(
    N: int,
    cls: list[tuple] = []
) -> NDArray:
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


def CircuitTopology(
    i:tuple[int,int],
    j:tuple[int,int]
) -> int:
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


def TopoRatio(
    cls:list[tuple],
    minLoopLength: int = 0,
    nextNeighbor: int = 0
) -> tuple[float,float,float]:
    '''
    Function to compute the conformation class ratios by the type 1 loop
    conformation convention, further explanations can be found in the Bachelor
    thesis
    circuti topology conformation:
        - 1: loops in series
        - 2: loops entangled
        - 3: loops parallel

    Input:
    cls (dtype = list[tuple])...    list of tuples, each tuple contains
                                    indices of cross-linked monomers, expects
                                    Tuples to be (x,y): x < y

    Output:
    p1 (dtype = float)...           ratio of conformation class 1
    p2 (dtype = float)...           ratio of conformation class 2
    p3 (dtype = float)...           ratio of conformation class 3
    '''
    cls = [cl for cl in cls if np.abs(cl[0] - cl[1]) >= minLoopLength]
    cls.sort(key=lambda x: x[0])
    L = len(cls)
    if L < 2:
        return 0, 0, 0
    loopcon = np.zeros((L, L), dtype=int)

    if 2*nextNeighbor > L:
        raise ValueError(
            'Input for "nextNeighbor" is too large for number of loops ' \
            'available in this polymer.'
        )

    for ind, _ in np.ndenumerate(loopcon):
        if np.abs(ind[0] - ind[1]) > nextNeighbor or ind[0] == ind[1]:
            continue
        i = cls[ind[0]]
        j = cls[ind[1]]
        loopcon[ind] = CircuitTopology(i, j)
    
    C = np.sum(loopcon != 0)*2      # total number of loop conformations
    p1 = np.sum(loopcon == 1)/C     # ratio of loops in series
    p2 = np.sum(loopcon == 2)/C     # ratio of entangled loops
    p3 = np.sum(loopcon == 3)/C     # ratio of parallel loops

    return p1, p2, p3