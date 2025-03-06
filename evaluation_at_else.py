import os
import sys
sys.path.append(os.path.dirname(__file__))
import library_MoChaTo as lib
import glob
import h5py


root_dir = r'C:\Users\jonas\OneDrive\TU Dresden\Physik-Studium, Bachelor\Abschlussarbeit'
search_crit = r'\**\*.hdf5'

system = 'windows'              # clearify operating systsem for file handling

filter_obj = 'swell_sqiso_key'
datagroups = [filter_obj, 'swell_sqiso']
plot_path = r'C:\Users\jonas\OneDrive\TU Dresden\Physik-Studium, Bachelor\Abschlussarbeit\plots'


for path in glob.glob(root_dir+search_crit, recursive=True):
    with h5py.File(path, 'r') as file:
        file.visit(lambda x: lib.filter_func(name=x, file=file,\
            filter_obj=filter_obj, get_datasets=datagroups, plot_path=plot_path, system=system, TestRun=True))