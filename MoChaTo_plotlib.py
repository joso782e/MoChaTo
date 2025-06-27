'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
MoChaTo: Library for plotting evaluation  data of SCNPs
-------------------------------------------------------------------------------
written by Jonas Soucek, 2025
at TU Dresden and Leibniz Institute of Polymer Research
'''


import json
import os
from os.path import exists
import sys
sys.path.append(os.path.dirname(__file__))
import data_evaluation.Scripts.MoChaTo_datalib as datalib
import numpy as np
import matplotlib.pyplot as plt


with open('.\\data_evaluation\\Scripts\\config.json', 'r') as configf:
    config = json.load(configf)


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
            dataobjs:list[datalib.FileData]
    ):
        self.rule = PlotRule(plotaspects=plotaspects)
        if isinstance(self.rule.ydata, str):
            self.rule.ydata = [self.rule.ydata]
        if (
            isinstance(self.rule.yerr, str)
            and self.rule.plot == 'errorbar'
        ):
                self.rule.yerr = [self.rule.yerr]

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
        
        for ydata in self.rule.ydata:
            if not (ydata in dir(dataobjs[0])):
                raise AttributeError(
                    f'Rule "ydata = {ydata}" not as attribute in '
                    'FileData objects. Please set ydata to one of the '
                    'following values:'
                    f'\n{dir(dataobjs[0])}'
                    '\n, specificly to "c1"/"c2" for coordinate 1/2 in '
                    'PC-space or to "PC1"/"PC2" for principal component 1/2'
                )
        
        if not (self.rule.sortby == 'N' or self.rule.sortby == 'f'):
            raise ValueError(
                f'Input "{self.rule.sortby}" for sortby not supported. '
                'Please set sortby to "N" or "f"'
            )        

        self.SetDataObjs(dataobjs=dataobjs)
    
    def SetDataObjs(self, dataobjs:list[datalib.FileData]) -> None:
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

    def GetData(self) -> None:
        '''
    
        '''
        rule = self.rule

        xdata = []      # list to store xdata
        ydata = []      # list to store ydata
        # if errorbar plot, create lists to store x and y errors
        if rule.plot == 'errorbar':
            xerr = []
            yerr = []
        labels = []     # list to store labels for data sets
        ls = []         # list to store line styles
        lw = []         # list to store line widths
        marker = []     # list to store marker types
        ms = []         # list to store marker sizes
        color = []      # list to store colors for data sets

        # create cyclers for line styles, line widths, marker types,
        # marker sizes and colors
        lscycler = Cycler(
            rule.ls if isinstance(rule.ls, list) else [rule.ls]
        )
        lwcycler = Cycler(
            rule.lw if isinstance(rule.lw, list) else [rule.lw]
        )
        markercycler = Cycler(
            rule.marker if isinstance(rule.marker, list) else [rule.marker]
        )
        mscycler = Cycler(
            rule.ms if isinstance(rule.ms, list) else [rule.ms]
        )
        colorcycler = Cycler(
            rule.color if isinstance(rule.color, list) else [rule.color]
        )

        for i in range(len(rule.ydata)):
            for obj in self.data:
                if not hasattr(obj, f'{rule.ydata[i]}scalfac'):
                    setattr(obj, f'{rule.ydata[i]}scalfac', obj.scalfac)
            for j in getattr(rule, f'{rule.sortby}rule'):
                # check if data set with current rule exists
                if len([
                    obj for obj in self.data
                    if getattr(obj, rule.sortby) == j
                ]) == 0:
                    continue
                # get xdata for current rule and append to xdata list
                xdata.append(
                    np.stack([
                        getattr(obj, rule.xdata) for obj in self.data
                        if getattr(obj, rule.sortby) == j
                    ], axis=0)
                )
                # get ydata for current rule and append to ydata list
                ydata.append(
                    np.stack([
                        getattr(obj, f'{rule.ydata[i]}scalfac')\
                        *getattr(obj, rule.ydata[i]) for obj in self.data
                        if getattr(obj, rule.sortby) == j
                    ], axis=0)
                )
                # if errorbar plot, get data for errors
                if rule.plot == 'errorbar':
                    # if xerr is given, get xerr for current rule and append to
                    # xerr list
                    if rule.xerr in dir(self.data[0]):
                        xerr.append(
                            np.stack([
                                getattr(obj, rule.xerr[i]) for obj in self.data
                                if getattr(obj, rule.sortby) == j
                            ], axis=0)
                        )
                    elif (
                        isinstance(rule.xerr[i], (int, float))
                        or str(rule.xerr[i]).isnumeric()
                    ):
                        xerr.append(np.full_like(
                            xdata[i], float(rule.xerr[i]), dtype=float
                        ))
                    else:
                        print(ValueError(
                           f'Warning! Input for xerr empty or not supported. '
                            'xerr will be set to 0'
                        ))
                        xerr.append(
                            np.full_like(xdata[i], 0.0, dtype=float)
                        )
                    # if yerr is given, get yerr for current rule and append to
                    # yerr list
                    if rule.yerr in dir(self.data[0]):
                        yerr.append(
                            np.stack([
                                getattr(obj, rule.yerr[i]) for obj in self.data
                                if getattr(obj, rule.sortby) == j
                            ], axis=0)
                        )
                    elif (
                        isinstance(rule.yerr[i], (int, float))
                        or str(rule.yerr[i]).isnumeric()
                    ):
                        yerr.append(np.full_like(
                            ydata[i], float(rule.yerr[i]),dtype=float
                        ))
                    else:
                        print(ValueError(
                           f'Warning! Input for xerr empty or not supported. '
                            'yerr will be set to 0'
                        ))
                        yerr.append(
                            np.full_like(ydata[i], 0.0, dtype=float)
                        )
                
                # create label for current rule and append to labels list
                if len(rule.ydata) > 1:
                    lstr = f'{rule.label[i]} at ${rule.sortby}={round(j,2)}$'
                else:
                    lstr = f'${rule.sortby}={round(j,2)}$'
                labels.append([
                    lstr for obj in self.data if getattr(obj, rule.sortby) == j
                ])
                # append line styles, line widths, types, marker sizes and
                # colors to corresponding lists
                ls.append(
                    [lscycler.Current()
                     if ydata[-1].ndim == 1 else lscycler.Current()
                     for _ in range(ydata[-1].shape[0])
                    ]
                )
                lw.append(
                    [lwcycler.Current()
                     if ydata[-1].ndim == 1 else lwcycler.Current()
                     for _ in range(ydata[-1].shape[0])
                    ]
                )
                marker.append(
                    [markercycler.Current()
                     if ydata[-1].ndim == 1 else markercycler.Current()
                     for _ in range(ydata[-1].shape[0])
                    ]
                )
                ms.append(
                    [mscycler.Current()
                     if ydata[-1].ndim == 1 else mscycler.Current()
                     for _ in range(ydata[-1].shape[0])
                    ]
                )
                color.append(
                    [colorcycler.Current()
                     if ydata[-1].ndim == 1 else colorcycler.Current()
                     for _ in range(ydata[-1].shape[0])
                    ]
                )
                # increase cyclers if length of rule attributes does not match
                # length of ydata
                if len(lscycler.iterable) != len(rule.ydata):
                    lscycler.Increase()
                if len(lwcycler.iterable) != len(rule.ydata):
                    lwcycler.Increase()
                if len(markercycler.iterable) != len(rule.ydata):
                    markercycler.Increase()
                if len(mscycler.iterable) != len(rule.ydata):
                    mscycler.Increase()
                if len(colorcycler.iterable) != len(rule.ydata):
                    colorcycler.Increase()
            # increase cyclers if length of rule attributes matches
            # length of ydata   
            if len(lscycler.iterable) == len(rule.ydata):
                lscycler.Increase()
            if len(lwcycler.iterable) == len(rule.ydata):
                lwcycler.Increase()
            if len(markercycler.iterable) == len(rule.ydata):
                markercycler.Increase()
            if len(mscycler.iterable) == len(rule.ydata):
                mscycler.Increase()
            if len(colorcycler.iterable) == len(rule.ydata):
                colorcycler.Increase()

        self.xdata = xdata
        self.ydata = ydata
        if rule.plot == 'errorbar':
            self.xerr = xerr
            self.yerr = yerr
        self.labels = labels
        self.ls = ls
        self.lw = lw
        self.marker = marker
        self.ms = ms
        self.color = color

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
            rule.plot + '_' + rule.plotdomain + '_' + str(rule.ydata) +\
            '_vs_'+ rule.xdata + seperator + f'N_{rule.Nrule}'
                
        fig = plt.figure(figsize=rule.figsize)      # create figure
        ax = fig.add_subplot(1, 1, 1)               # create subplot

        # set title and axis labels
        ax.set_title(rule.title)
        ax.set_xlabel(rule.xlabel)
        ax.set_ylabel(rule.ylabel)
        ax.grid(axis='both', ls='--', lw=0.5, color='grey')

        xdata = self.xdata
        ydata = self.ydata
        if rule.plot == 'errorbar':
            xerr = self.xerr
            yerr = self.yerr
        labels = self.labels
        ls = self.ls
        lw = self.lw
        marker = self.marker
        ms = self.ms
        color = self.color


        for i in range(len(xdata)):
            # asume more data points then samples
            if xdata[i].shape[0] < xdata[i].shape[1]:
                xdata[i] = xdata[i].T
                if rule.plot == 'errorbar':
                    xerr[i] = xerr[i].T
            # check if one axis lenght of ydata[i] matches length of axis 0 in
            # xdata[i], if not raise ValueError
            if xdata[i].shape[0] != ydata[i].shape[0]:
                ydata[i] = ydata[i].T
                if rule.plot == 'errorbar':
                    yerr[i] = yerr[i].T
                if xdata[i].shape[0] != ydata[i].shape[0]:
                    raise ValueError(
                        f'Data set {i+1} can not be plotted against each '
                        f'other because their shapes {xdata[i].shape} and '
                        f'{ydata[i].shape} can not be broadcast together.'
                    )
            ax.set_prop_cycle(
                color=color[i], linestyle=ls[i], lw=lw[i],
                marker=marker[i], ms=ms[i]
            )
                        
            # plot data as diagram or histogram
            if rule.plot == 'diag':
                # dicide wether to turn auto scale on or not
                xlow = rule.xlim[0]
                xup = rule.xlim[1]
                ylow = rule.ylim[0]
                yup = rule.ylim[1]
                if xup or xlow:
                    xauto = False
                else:
                    xauto = True
                
                if yup or ylow:
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
                        xdata[i], ydata[i], label=labels[i]
                    )
                else:
                    # plot data in normal plot
                    ax.plot(
                        xdata[i], ydata[i], label=labels[i]
                    )
                    ax.set_xscale(rule.xscale)
                    ax.set_yscale(rule.yscale)
            elif rule.plot == 'errorbar':
                ax.errorbar(
                    np.squeeze(xdata[i]), np.squeeze(ydata[i]),
                    np.squeeze(yerr[i]), np.squeeze(xerr[i]), label=labels[i]
                )
            elif rule.plot == 'hist':
                # create bins and plot data in histogram
                bins = np.linspace(
                    np.min(ydata[i]), np.max(ydata[i]), rule.binnum
                )
                ax.hist(
                    ydata[i].flatten(), bins=bins, density=True,
                    label=labels
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



    
def save_plot(
        fig:plt.Figure, name:str, path:str, system:str=config['system'],
        fileformat:str=config['fileformat']
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