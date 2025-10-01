import os
from os.path import exists
import sys
sys.path.append(os.path.dirname(__file__))
import MoChaTo_datalib as datalib
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import networkx as nx


class PlotData:
    def __init__(
            self,
            rules: dict,
            datalist: list
        ):
        '''
        Define constructor of class 'PlotData'

        Input:
        rules (dtype = dict)...         dictionary of key-value pairs that
                                        define the plot
        datalist (dtype = list)...      list with objects that contain data to
                                        be plotted
        '''
        if not isinstance(rules, dict):
            raise TypeError(
                f"Type {type(rules)} of input 'rules' not accepted! Input " \
                "'rules' must be a dictionary."
            )
        
        if not isinstance(datalist, list):
            raise TypeError(
                f"Type {type(rules)} of input 'datalist' not accepted! " \
                "Input 'datalist' must be a list."
            )
        
        rules = self.CheckRules(rules=rules)
        self.rules = rules
        self.datalist = datalist

    @staticmethod
    def CheckRules(
        rules: dict
    ) -> dict:
        '''
        Method to check wether input dictionary contains necessary keys
        '''
        if not "xdata" in rules.keys():
            raise KeyError(
                "Warning! Input dictionary 'rules' must contain key 'xdata' " \
                "in order to proceed! The value of key 'xdata' must be a " \
                "string corresponding to the attribute of the objects in " \
                "input 'datalist' that contain x-values of the data to be " \
                "plotted."
            )
        
        if not "ydata" in rules.keys():
            raise KeyError(
                "Warning! Input dictionary 'rules' must contain key 'ydata' " \
                "in order to proceed! The value of key 'ydata' must be a " \
                "string corresponding to the attribute of the objects in " \
                "input 'datalist' that contain y-values of the data to be " \
                "plotted."
            )
        
        if not "plot" in rules.keys():
            print(
                "Warning! Input dictionary 'rules' does not contain key " \
                "'plot'. The key is added to 'rules' and the string " \
                "'diag' is assigned."
            )
            rules['plot'] = "diag"
        
        if not "representation" in rules.keys():
            print(
                "Warning! Input dictionary 'rules' does not contain key "
                "'representation'. The key is added to 'rules' and the " \
                "string 'direct' is assigned."
            )
            rules['representation'] = "direct"
        
        if not "xscale" in rules.keys():
            print(
                "Warning! Input dictionary 'rules' does not contain key " \
                "'xscale'. The key is added to 'rules' and the string " \
                "'linear' is assigned."
            )
            rules['xscale'] = "linear"

        if not "yscale" in rules.keys():
            print(
                "Warning! Input dictionary 'rules' does not contain key " \
                "'yscale'. The key is added to 'rules' and the string " \
                "'linear' is assigned."
            )
            rules['yscale'] = "linear"

        if not 'labels' in rules.keys():
            print(
                "Warning! Input dictionary 'rules' does not contain key " \
                "'labels'. Graph labels will be ['data1', 'data2', ...]."
            )

        return rules
    
    @staticmethod
    def CheckDataArrays(
        xData: np.ndarray,
        yData: np.ndarray
    ) -> None:
        '''
        Method to check whether input data arrays have the same shape
        '''
        xData = np.array(xData)
        yData = np.array(yData)

        if yData.size[0] != xData.size[0]:
            yData = yData.T
            if yData.size[0] != xData.size[0]:
                raise ValueError(
                    f"Warning! The shape of xData {xData.shape} and " \
                    f"yData {yData.T.shape} can not be broadcasted " \
                    "together. Please check the data objects in " \
                    "input 'datalist'."
                )
        return xData, yData

    def CreateFigure(
        self,
        title: str = None,
        xLabel: str = None,
        yLabel: str = None,
        figSize: tuple[float, float] | None = None,
        xLim: tuple[float, float] | None = None,
        yLim: tuple[float, float] | None = None
    ) -> None:
        '''
        
        '''
        rules = self.rules

        # create figure and axis handler
        fig = plt.figure(figsize=figSize)
        ax = fig.add_subplot(111)

        # determine axis limits and dicide wether to turn auto scale on or not
        if xLim:
            xLow, xUp = xLim[0], xLim[1]
            xAuto = False
        else:
            xLow, xUp = None, None
            xAuto = True

        if yLim:
            yLow, yUp = yLim[0], yLim[1]
            yAuto = False
        else:
            yLow, yUp = None, None
            yAuto = True

        # set axis view limits
        ax.set_xlim(left=xLow, right=xUp, auto=xAuto)
        ax.set_ylim(bottom=yLow, top=yUp, auto=yAuto)

        # set title
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f'{rules['ydata']} vs. {rules['xdata']}')

        # set axis labels
        if xLabel:
            ax.set_xlabel(xLabel)
        else:
            ax.set_xlabel(f'{rules['xdata']}')
        if yLabel:
            ax.set_ylabel(yLabel)
        else:
            ax.set_ylabel(f'{rules['ydata']}')

        # set grid
        ax.grid(axis='both', ls='--', lw=0.5, color='grey')

        if rules['representation'] == 'Kratky':
            ax.set_xscale('log')
            ax.set_yscale('log')
        else:
            ax.set_xscale(rules['xscale'])
            ax.set_yscale(rules['yscale'])

        self.fig = fig
        self.ax = ax
    
    def GetData(
        self,
        sort: str | None = None
    ):
        '''
        
        '''
        rules = self.rules
        dataList = self.datalist

        self.labels = []
        
        if sort:
            # sort dataobjects for same value in property defined by input
            # 'sort'
            sortVals = np.unique(
                [getattr(obj, sort) for obj in dataList]
            )
            for val in sortVals:
                xData = [
                    getattr(obj, rules['xdata']) for obj in dataList
                    if getattr(obj, sort) == val
                ]
                yData = [
                    getattr(obj, rules['ydata']) for obj in dataList
                    if getattr(obj, sort) == val
                ]
                self.labels.append(f'{sort} = {val}')
                xData, yData = self.CheckDataArrays(xData, yData)
                yield(xData, yData)
        else:
            if 'labels' in rules.keys():
                self.labels = rules['labels']
            else:
                i = 0

            for obj in dataList:
                xData = getattr(obj, rules['xdata'])
                yData = getattr(obj, rules['ydata'])
                if not ('labels' in rules.keys()):
                    self.labels.append(f'data{i+1}')
                    i += 1
                    
                xData, yData = self.CheckDataArrays(xData, yData)
                yield(xData, yData)

    def AddData2Plot(
        self,
        sort: str | None = None
    ) -> None:
        '''
        
        '''
        rules = self.rules
        ax = self.ax

        labelCycler = Cycler(self.labels)
        

        for xData, yData in self.GetData(sort=sort):
            if rules['representation'] == 'Kratky':
                yData = np.multiply(yData, xData**2)

            if rules['plot'] == 'statistical binning':
                xData, yData, xErr, yErr, xScatter, yScatter = BinData(
                    xData, yData, rules['bins']
                )
                xMin, xMax, yPos = ComputeErrorbarLimits(
                    xData, yData, xErr, 'x'
                )
                yMin, yMax, xPos = ComputeErrorbarLimits(
                    xData, yData, yErr, 'y'
                )

                # include data for error bar plotting to data of boundary bar
                # plotting
                yPos = np.concatenate([yPos, yData])
                xMin = np.concatenate([xMin, xData - xErr])
                xMax = np.concatenate([xMax, xData + xErr])
                xPos = np.concatenate([xPos, xData])
                yMin = np.concatenate([yMin, yData - yErr])
                yMax = np.concatenate([yMax, yData + yErr])

                vLineHandle = ax.vlines(
                    xPos, ymin=yMin, ymax=yMax, color='orangered', lw=0.5
                )
                hLineHandle = ax.hlines(
                    yPos, xmin=xMin, xmax=xMax, color='orangered', lw=0.5
                )

                xFill, upperFill, lowerFill = ComputeScatterTube(
                    xData, yData, xScatter, yScatter
                )
                fillHandle = ax.fill_between(
                    xFill, upperFill, lowerFill, color='orange', alpha=0.4
                )
            
            dataHandle = ax.plot(
                xData, yData
            )

            if rules['plot'] == 'statistical binning':
                handles = [(dataHandle, vLineHandle, hLineHandle), fillHandle]
                labels = [labelCycler.Cycle(), 'data scattering']
            else:
                handles = dataHandle
                labels = labelCycler.Cycle()

            ax.legend(
                handles, labels
            )


class PlotSCNP:
    '''
    Class to plot a graph modeling the cross-linked conformation of a SCNP in
    2D
    '''

    def __init__(
        self,
        name: str,
        sequence: np.ndarray,
        crosslinks: np.ndarray,
        path: str
    ):
        '''
        Constructor for PlotSCNP class

        Input:
        name (dtype = str)...               name of plot file
        sequence (dtype = nd.ndarray)...    1D-array object with SCNP
                                            data
        crosslinks (dtype = nd.ndarray)...  2D-array with crosslinks other
                                            than those forming precursors
        path (dtype = str)...               path to save plot file
        '''
        self.name = name
        self.sequence = sequence
        self.crosslinks = crosslinks
        self.path = path
        self.fig = plt.figure(figsize=(6, 6))
        self.ax = self.fig.add_subplot(1, 1, 1)

    def ConstructGraph(self) -> None:
        '''
        
        '''
        seq = self.sequence
        cl = self.crosslinks

        if cl.ndim != 2:
            raise ValueError(
                'Crosslinks must be a 2D-array with shape (2, n) or (n, 2), '
                f'got {cl.shape} instead.'
            )
        
        if cl.shape[0] != 2:
            cl = cl.T

        self.G = nx.Graph()                 # create empty graph object
        spec = ['r', 'b', 'a']
        color = ['orangered', 'black', 'limegreen']

        nodes = [(i, {'color': color[seq[i]-2]}) for i in range(len(seq))]
        edges = [
            (i, i+1, {'color': 'black', 'ls': '-'}) for i in range(len(seq)-1)
        ]
        edges += [
            (cl[0, i], cl[1, i], {'color': 'cyan', 'ls': '--'})
            for i in range(cl.shape[1]) if (cl[0, i] and cl[1, i])
        ]

        self.labels = {
            i: f'{spec[seq[i]-2]}' for i in range(len(seq))
        }

        self.G.add_nodes_from(nodes)
        self.G.add_edges_from(edges)

    def DrawGraph(self):
        '''

        '''
        if not hasattr(self, 'G'):
            self.ConstructGraph()
        
        node_color = [self.G.nodes[i]['color'] for i in self.G.nodes]
        edge_color = [self.G[i][j]['color'] for (i,j) in self.G.edges]
        edge_ls = [self.G[i][j]['ls'] for (i,j) in self.G.edges]
        
        pos = nx.kamada_kawai_layout(self.G)
        nx.draw(
            self.G, pos, ax=self.ax, labels=self.labels, node_size=25,
            node_color=node_color, edge_color=edge_color, style=edge_ls,
            with_labels=False
        )
        self.ax.legend(
            handles=[
                plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor='orangered', markersize=7,
                    label='reactive bead'
                ),
                plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor='black', markersize=7,
                    label='backbone bead'
                ),
                plt.Line2D(
                    [0], [0], marker='o', color='w',
                    markerfacecolor='limegreen', markersize=7,
                    label='activated bead'
                ),
                plt.Line2D(
                    [0], [0], color='cyan', lw=1, ls='--',
                    label='crosslink'
                ),
                plt.Line2D(
                    [0], [0], color='black', lw=1, label='single-chain bond'
                )
            ]
        )

        SavePlot(
            fig=self.fig, name=self.name, path=self.path
        )


class Cycler:
    '''
    Class to create a cycler
    '''

    def __init__(self, iterable):
        self.count = 0
        if not isinstance(iterable, (list, tuple, np.ndarray)):
            raise TypeError(
                f'Warning! Input of type {type(iterable)} not supported.'
            )
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
    
    def Current(self):
        '''
        Function to get current item in iterable
        '''
        if self.count >= len(self.iterable):
            self.count = 0
        return self.iterable[self.count]
    
    def Increase(self):
        '''
        Function to increase count by 1
        '''
        if self.count >= len(self.iterable):
            self.count = 0
        else:
            self.count += 1
    
    def FullCycle(self):
        '''
        Function to perform full cycle through iterable
        '''
        for i in range(len(self.iterable)):
            yield self.Cycle()
    
    def Reset(self):
        '''
        Function to reset cycler
        '''
        self.count = 0


def BinData(
    xData: np.ndarray,
    yData: np.ndarray,
    bins: np.ndarray | int = 20,
    ppb: bool = False,
    xLower: float | None = None,
    xUpper: float | None = None
) -> tuple[np.ndarray]:
    '''
    Function to bin ydata along given xdata axis, calculate their mean and
    variance for each bin seperately

    
    Input:
    xData (dtype = array-like)
        x-values

    yData (dtype = array-like)
        y-values

    bins (dtype = float or array-like)...
        - if int: sets number of evenly spaced bins
        - if array-like: sets bin edges

    ppb (dtype = bool)...
        if True, computes bins so that equal number of data points per bin
        using bins as integer

        
    Output:
    x
        mean x-values for each bin

    y
        mean y-values for each bin

    xmeanerr
        error of mean x-values for each bin

    ymeanerr
        error of mean y-values for each bin

    xscatter
        standard deviation of x-values for each bin

    yscatter
        standard deviation of y-values for each bin
    '''
    diff = np.max(xData) - np.min(xData)

    xMin = xLower if xLower else np.min(xData) - 0.001*diff
    xMax = xUpper if xUpper else np.max(xData) + 0.001*diff

    if ppb:
        if not isinstance(bins, int):
            raise TypeError(
                'If ppb is True, bins must be an integer specifying the '
                'number of data points per bin.'
            )
        sort = np.argsort(xData)
        sort = xData[sort]
        x = sort[::bins]
    else:
        if isinstance(bins, int):
            x = np.linspace(xMin, xMax, bins+1)
        else:
            x = bins

    x[0] = xMin
    x[-1] = xMax

    y = np.zeros_like(x[1:])
    xMeanErr = np.zeros_like(x[1:])
    yMeanErr = np.zeros_like(x[1:])
    xScatter = np.zeros_like(x[1:])
    yScatter = np.zeros_like(x[1:])

    for i in range(len(x[1:])):
        binned = yData[xData >= x[i]]
        buffer = xData[xData >= x[i]]
        binned = binned[buffer <= x[i+1]]
        buffer = buffer[buffer <= x[i+1]]
        if len(binned):
            y[i] = np.mean(binned)
            if len(binned) >= 10:
                yScatter[i] = np.sqrt(np.var(binned, ddof=1))
                xScatter[i] = np.sqrt(np.var(buffer, ddof=1))
            else:
                yScatter[i] = (max(binned) - min(binned))/2
                xScatter[i] = (max(buffer) - min(buffer))/2
            yMeanErr[i] = yScatter[i]/np.sqrt(len(binned))
            xMeanErr[i] = xScatter[i]/np.sqrt(len(binned))

    idx0 = np.nonzero(y)
    x = (x[1:] + x[:-1])/2

    return np.array(x[idx0]), \
        np.array(y[idx0]), \
        np.array(xMeanErr[idx0]), \
        np.array(yMeanErr[idx0]), \
        np.array(xScatter[idx0]), \
        np.array(yScatter[idx0])


def ComputeErrorbarLimits(
    xData: np.ndarray,
    yData: np.ndarray,
    err: np.ndarray,
    forError: str
):
    '''
    
    '''
    if forError == 'y':
        errDirect = yData
        lengthData = xData
    elif forError == 'x':
        errDirect = xData
        lengthData = yData
    else:
        raise ValueError(
            f'Warning! Input {forError} for "forError" not supported. '
            'Choose either "x" or "y".'
        )

    diff = np.max(lengthData) - np.min(lengthData)
    pos = np.concatenate(
        [errDirect + err, errDirect - err]
    )
    minbar = lengthData - diff/100
    minbar = np.concatenate([minbar, minbar])
    maxbar = lengthData + diff/100
    maxbar = np.concatenate([maxbar, maxbar])

    return minbar, maxbar, pos


def ComputeScatterTube(
    xData: np.ndarray,
    yData: np.ndarray,
    stdDevX: np.ndarray,
    stdDevY: np.ndarray
) -> tuple[np.ndarray]:
    '''
    Function to compute the scatter tube for given x and y data with errors

    
    Input:
    xData (dtype = array-like)
        x-values

    yData (dtype = array-like)
        y-values

    stdDevX (dtype = array-like)
        errors in x-values

    stdDevY (dtype = array-like)
        errors in y-values

        
    Output:
    xFill
        x-values for scatter tube

    upperY
        upper y-values for scatter tube
    
    lowerY
        lower y-values for scatter tube
    '''
    x1 = xData + stdDevX
    x2 = xData - stdDevX
    xFill = np.concatenate([x1, x2])
    sortIdx = np.argsort(xFill)
    upperY = yData + stdDevY
    upperY = np.concatenate([upperY, upperY])
    lowerY = yData - stdDevY
    lowerY = np.concatenate([lowerY, lowerY])

    return xFill[sortIdx], upperY[sortIdx], lowerY[sortIdx]


def SavePlot(
    fig: plt.Figure,
    name: str,
    path: str,
    system: str = 'windows',
    fileformat: str = '.svg'
) -> None:
    '''
    Function to safe plot

    Input:
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


def SCNPFilter(
    dataobj: datalib.FileData,
    N: int,
    constrains: dict,
    condition: list[str]
) -> np.ndarray:
    '''
    Function to filter SCNP data objects

    Input:
    dataobj (dtype = datalib.FileData)...   FileData object with SCNP data
    N (dtype = int)...                      number of SCNPs to check
    constrains (dtype = dict)...            dictionary with constrains for
                                            filtering dataobj, keys are
                                            attributes of FileData object,
                                            must match length of condition
    condition (dtype = list[str])...        condition if value in constrain
                                            category of SCNP should be 'less',
                                            'equal' or 'greater' then the
                                            constrains value,
                                            must match length of constrains

    Output:
    idx (dtype = np.ndarray)...             boolean array with True for
                                            values that match the constrains
                                            and condition, False otherwise
    '''
    conskey = list(constrains.keys())    # get keys of constrains dict

    if len(conskey) != len(condition):
        if len(condition) == 1:
            condition = [condition[0]] * len(conskey)
            print(condition)
        else:
            raise ValueError(
                'Length of constrains and condition must match. '
                f'Length of constrains: {len(conskey)}, '
                f'length of condition: {len(condition)}'
            )
    
    idx = np.full(N, 0)
    for i in range(len(conskey)):
        param = getattr(dataobj, conskey[i])  # get parameter from dataobj
        if condition[i] == 'less':
            # check if parameter is less then constrains value
            idx += param < constrains[conskey[i]]
        elif condition[i] == 'equal':
            # check if parameter is equal to constrains value
            idx += param == constrains[conskey[i]]
        elif condition[i] == 'greater':
            # check if parameter is greater then constrains value
            idx += param > constrains[conskey[i]]
        else:
            raise ValueError(
                f'Condition "{condition[i]}" not supported. '
                'Please use "less", "equal" or "greater".'
            )

    # return boolean array with True for values that match all condition
    return idx == len(conskey)


def FilterPlotSCNP(
    dataobj:datalib.FileData, constrains:dict, condition:list[str],
    SCNPpath:str
) -> None:
    '''
    
    '''
    idx = SCNPFilter(
            dataobj=dataobj, N=dataobj.S.shape[0], constrains=constrains,
            condition=condition,
        )
    
    seq = dataobj.sequence[idx,:]
    cl1 = dataobj.clmat[idx,:,1]
    cl2 = dataobj.clmat[idx,:,2]
    name = []

    for i in range(len(idx)):
        if idx[i]:
            nstr =''
            for conkey in constrains.keys():
                atr = getattr(dataobj,conkey)
                nstr += f'{conkey}_{round(float(atr[i]),5)}'
            name.append(nstr)
        
    for i in range(len(name)):
        outliercl = np.stack(
            [cl1[i,:], cl2[i,:]], axis=1
        )
        scnp = PlotSCNP(
            name=name[i], sequence=seq[i,:],
            crosslinks=outliercl,
            path=SCNPpath
        )
        scnp.DrawGraph()