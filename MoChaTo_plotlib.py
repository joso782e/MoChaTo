'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
MoChaTo: Library for plotting evaluation  data of SCNPs
-------------------------------------------------------------------------------
written by Jonas Soucek, 2025
at TU Dresden and Leibniz Institute of Polymer Research

Library of support classes and functions mainly for plotting data
'''


import json
import os
from os.path import exists
import sys
sys.path.append(os.path.dirname(__file__))
import MoChaTo_datalib as datalib
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import networkx as nx


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
        self.legend = False
        self.legend_loc = 'upper right'
        self.label = None

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
        if self.rule.plot == 'errorbar':
            if isinstance(self.rule.yerr, str):
                self.rule.yerr = [self.rule.yerr]
            if isinstance(self.rule.xerr, str):
                self.rule.xerr = [self.rule.xerr]

        if not (
            set(self.rule.Nrule).issubset([obj.N for obj in dataobjs]) or
            self.rule.Nrule == 'all'
        ):
            raise ValueError(
                f'Rule "N = {self.rule.Nrule}" not subset of all N values. '
                 'Please set Nrule to one or a list of the following values:'
                f'\n{np.unique([obj.N for obj in dataobjs])}'
                 '\nor to "all"'
            )
        
        if not (
            set(self.rule.frule).issubset([obj.f for obj in dataobjs]) or
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
            rule.Nname = 'all'
        else:
            rule.Nname = rule.Nrule

        if rule.frule == 'all':
            rule.frule = np.unique([obj.f for obj in dataobjs])
            rule.fname = 'all'
        else:
            rule.fname = [round(f,2) for f in rule.frule]

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
                        getattr(obj, rule.ydata[i])
                        for obj in self.data if getattr(obj, rule.sortby) == j
                    ], axis=0)
                )
                # if errorbar plot, get data for errors
                if rule.plot == 'errorbar':
                    # if xerr is given, get xerr for current rule and append to
                    # xerr list
                    if rule.xerr[i] in dir(self.data[0]):
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
                        xerr.append(float(rule.xerr[i]))
                    else:
                        print(
                           f'Warning! Input for xerr empty or not supported. '
                            'xerr will be set to 0'
                        )
                        xerr.append(0.0)
                    # if yerr is given, get yerr for current rule and append to
                    # yerr list
                    if rule.yerr[i] in dir(self.data[0]):
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
                        yerr.append(rule.yerr[i])
                    else:
                        print(
                           f'Warning! Input for yerr empty or not supported. '
                            'yerr will be set to 0'
                        )
                        yerr.append(0.0)
                
                # create label for current rule and append to labels list
                if rule.label:
                    if len(rule.ydata) > 1:
                        lstr = f'{rule.label[i]}'
                    else:
                        lstr = f'${rule.sortby}={round(j,2)}$' if rule.sortby == 'f' else f'${rule.sortby}={round(j,2)}$'
                    labellist = [
                        lstr for obj in self.data
                        if getattr(obj, rule.sortby) == j
                    ]
                    if len(labellist) > 1:
                        labels.append(np.unique(labellist))
                    else:
                        labels.append(lstr)
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
    
    def Quatsch(
            self, eva_path:str=config['eva_path'], system:str=config['system']
    ) -> None:
        
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
        self.figpath = eva_path + seperator + 'plots' + seperator +\
            rule.plot + '_' + rule.plotdomain + '_' + str(rule.ydata) +\
            '_vs_'+ rule.xdata + seperator + f'N_{rule.Nname}'
                
        fig = plt.figure(figsize=rule.figsize)      # create figure
        ax = fig.add_subplot(1, 1, 1)               # create subplot

        # set title and axis labels
        ax.set_title(rule.title)
        ax.set_xlabel(rule.xlabel)
        ax.set_ylabel(rule.ylabel)
        ax.grid(axis='both', ls='--', lw=0.5, color='grey')

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

        self.ax = ax
        self.fig = fig

        for sort in getattr(self, f'{rule.sortby}rule'):
            for obj in [
                obj for obj in self.data
                if getattr(obj, f'{rule.sortby}') == sort
            ]:
                self.PlotData(obj)

    def PlotData(self, obj:list[datalib.FileData]) -> None:
        '''
        
        '''
        rule = self.rule
        xdata = rule.xdata
        ax = self.ax

        if len(obj) == 1:
            for ydata in rule.ydata:
                ax.plot()


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
        self.figpath = eva_path + seperator + 'plots' + seperator +\
            rule.plot + '_' + rule.plotdomain + '_' + str(rule.ydata) +\
            '_vs_'+ rule.xdata + seperator + f'N_{rule.Nname}'
                
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
            # plot data as diagram or histogram
            if rule.plot == 'diag':
                    # asume more data points then samples
                if xdata[i].ndim > 1:
                    if xdata[i].shape[0] < xdata[i].shape[1]:
                        xdata[i] = xdata[i].T
                        if rule.plot == 'errorbar':
                            if not isinstance(xerr[i], float):
                                xerr[i] = xerr[i].T
                # check if one axis lenght of ydata[i] matches length of axis
                # 0 in xdata[i], if not raise ValueError
                if xdata[i].shape[0] != ydata[i].shape[0]:
                    ydata[i] = ydata[i].T
                    if rule.plot == 'errorbar':
                        if not isinstance(yerr[i], float):
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

                # make xdata and ydata plot compattible
                if ydata[i].ndim == xdata[i].ndim + 1:
                    ydata[i] = np.squeeze(ydata[i])
                    xdata[i] = np.full_like(ydata[i], xdata[i])

                if rule.plotdomain == 'Kratky':
                    # plot data in Kratky plot
                    ydata[i] = ydata[i]*xdata[i]**2
                    if rule.label:
                        ax.loglog(
                            xdata[i], ydata[i], label=labels[i]
                        )
                    else:
                        ax.loglog(xdata[i], ydata[i])
                else:
                    # plot data in normal plot
                    if rule.label:
                        ax.plot(
                            xdata[i], ydata[i], label=labels[i]
                        )
                    else:
                        ax.plot(xdata[i], ydata[i])
                    ax.set_xscale(rule.xscale)
                    ax.set_yscale(rule.yscale)

            elif rule.plot == 'errorbar':
                ax.set_prop_cycle(
                    color=color[i], linestyle=ls[i], lw=lw[i],
                    marker=marker[i], ms=ms[i]
                )
                ax.set_xscale(rule.xscale)
                ax.set_yscale(rule.yscale)
                if rule.label:
                    ax.errorbar(
                        np.squeeze(xdata[i]), np.squeeze(ydata[i]),
                        np.squeeze(yerr[i]), np.squeeze(xerr[i]),
                        ecolor='orangered', elinewidth=0.5,
                        label=labels[i]
                    )
                else:
                    ax.errorbar(
                        np.squeeze(xdata[i]), np.squeeze(ydata[i]),
                        np.squeeze(yerr[i]), np.squeeze(xerr[i]),
                        ecolor='orangered', elinewidth=0.5
                    )
                if np.any(yerr[i]):
                    minx, maxx, posy = PlotErrorbarLimits(
                        xdata[i], ydata[i], err=yerr[i], forerror='y'
                    )
                    ax.hlines(
                        posy, minx, maxx, color='orangered', linewidth=0.5
                    )
                if np.any(xerr[i]):
                    miny, maxy, posx = PlotErrorbarLimits(
                        xdata[i], ydata[i], err=xerr[i], forerror='x'
                    )
                    ax.vlines(
                        posx, miny, maxy, color='orangered', linewidth=0.5
                    )

            elif rule.plot == 'hist':
                # compute histogram
                counts, bins = np.histogram(ydata[i], bins=rule.binnum)
                counts = counts/np.sum(counts)
                bins = (bins[:-1] + bins[1:])/2

                if rule.plotdomain == 'expdistr':
                    counts = np.cumsum(counts)
                    def CumDistrFunc(x, a0, a1):
                        return a1 - np.exp(-a0*x)
                    fit, _ = opt.curve_fit(CumDistrFunc, bins, counts)
                    ax.set_ylim((None, 1.25))
                    ax.plot(
                        bins, CumDistrFunc(bins, fit[0], fit[1]), ls='-',
                        lw=1.0, color='orangered',
                        label=f'fitted cumulative\ndistribution function'
                    )
                    print(f'a0 = {fit[0]}')
                    print(f'a1 = {fit[1]}')
                    ax.legend(loc='upper left')
                else:
                    mean = np.sum(counts*bins)
                    # display mean value and calculate optimal position for
                    # text and line
                    ymax = np.max(counts)
                    ax.set_ylim((None, ymax*1.2))
                    ax.axvline(
                        mean, 0, 1/1.225, ls='--', lw=0.5, color='black'
                    )
                    ax.text(np.mean(
                        ydata[i]), ymax,
                        f'expectation value:\n{round(np.mean(ydata[i]),4)}',
                        horizontalalignment='center',
                        verticalalignment='bottom'
                    )

                ax.bar(bins, counts, color='dodgerblue')
                
            else:
                raise ValueError(
                    f'Plot type "{rule.plot}" not supported. '
                    'Please set plot to "diag", "hist" or "errorbar".'
                )
            
        if rule.legend:
            ax.legend(loc=rule.legend_loc)
            # remove duplicate legend entries
            # get legend handles and labels, create dictionary and replot
            # legend
            # source: (
            #   https://stackoverflow.com/questions/13588920/
            #   stop-matplotlib-repeating-labels-in-legend
            # )
            handles, labels = plt.gca().get_legend_handles_labels()
            labeldict= dict(zip(labels, handles))
            ax.legend(
                labeldict.values(), labeldict.keys(), loc=rule.legend_loc
            )
        
        self.ax = ax
        self.fig = fig
        
    def SavePlot(self, name:str=False, path:str=False) -> None:
        '''
        
        '''
        rule = self.rule
        if name:
            if path:
                SavePlot(
                    fig=self.fig, name=f'{name}', path=path
                )
            else:
                SavePlot(
                    fig=self.fig, name=f'{name}', path=self.figpath
                )
        else:
            if path:
                SavePlot(
                    fig=self.fig, name=f'f_{rule.fname}', path=path
                )
            else:
                SavePlot(
                    fig=self.fig, name=f'f_{rule.fname}', path=self.figpath
                )


class PlotSCNP:
    '''
    Class to plot a graph modeling the cross-linked conformation of a SCNP in
    2D
    '''

    def __init__(
        self, name:str, sequence:np.ndarray, crosslinks:np.ndarray, path:str
    ):
        '''
        Constructor for PlotSCNP class

        Input:
        sequence (dtype = nd.ndarray)...    1D-array object with SCNP
                                            data
        crosslinks (dtype = nd.ndarray)...  2D-array with crosslinks
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
            fig=self.fig, name=self.name, path=self.path)

    
def SavePlot(
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


def SCNPFilter(
        dataobj:datalib.FileData, N:int, constrains:dict, condition:list[str]
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


def PlotErrorbarLimits(
            xdata:np.ndarray, ydata:np.ndarray, err:np.ndarray, forerror:str
        ):
        '''
        
        '''
        if forerror == 'y':
            errdirect = ydata
            lengthdata = xdata
        elif forerror == 'x':
            errdirect = xdata
            lengthdata = ydata
        else:
            raise ValueError(
                f'Warning! Input {forerror} for "forerror" not supported. '
                'Choose either "x" or "Y".'
            )

        diff = np.max(lengthdata) - np.min(lengthdata)
        pos = np.concatenate(
            [errdirect + err, errdirect - err]
        )
        minbar = lengthdata - diff/100
        minbar = np.concatenate([minbar, minbar])
        maxbar = lengthdata + diff/100
        maxbar = np.concatenate([maxbar, maxbar])

        return minbar, maxbar, pos