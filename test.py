import os
import sys
sys.path.append(os.path.dirname(__file__))
import h5py
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import MoChaTo_plotlib as plotlib

class test:
    def __init__(self, func, N, f):
        self.x = np.linspace(0, 10, 100)
        self.x = np.concatenate([
            self.x + np.random.normal(0, 0.1, 100),
            self.x + np.random.normal(0, 0.1, 100),
            self.x + np.random.normal(0, 0.1, 100)
        ])
        self.y = func(self.x)
        self.y = self.y + np.random.normal(0, 0.1, 300)
        self.N = N
        self.f = f

a = test(np.sin, 100, 1)
b = test(np.cos, 100, 2)

data = [a, b]

rules = {
    'xdata': 'x',
    'ydata': 'y',
    'xlabel': 'X-axis',
    'ylabel': 'Y-axis',
    'labels': ['sin(x)', 'cos(x)'],
    'ls': ['-', '--'],
    'color': ['blue', 'red'],
    'plot': 'statistical binning'
}

plot = plotlib.PlotData(rules, data)
plot.CreateFigure(title='Sine and Cosine Functions', figSize=(8, 6))
plot.AddData2Plot(sort='f')

plt.show()