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
                    f"yData {yData.shape} can not be broadcasted " \
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
                xData, yData = self.CheckDataArrays(xData, yData)
                yield(xData, yData)
        else:
            for obj in dataList:
                xData = getattr(obj, rules['xdata'])
                yData = getattr(obj, rules['ydata'])
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

        for xData, yData in self.GetData(sort=sort):

            if rules['plot'] == 'errorbar':
                xData, yData, xErr, yErr, xScatter, yScatter = BinData(
                    xData, yData, rules['bins']
                )
                xMin, xMax, xPos = ComputeErrorbarLimits(
                    xData, yData, xErr, 'x'
                )
                yMin, yMax, yPos = ComputeErrorbarLimits(
                    xData, yData, yErr, 'y'
                )
                xFill, upperFill, lowerFill = ComputeScatterTube(
                    xData, yData, xScatter, yScatter
                )
                ax.fill_between(
                    xFill, upperFill, lowerFill, color='orange', alpha=0.4,
                    label='data scattering'
                )
                ax.

            if rules['representation'] == 'Kratky':
                yData = np.multiply(yData, xData**2)


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
            xData:np.ndarray,
            yData:np.ndarray,
            err:np.ndarray,
            forError:str
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