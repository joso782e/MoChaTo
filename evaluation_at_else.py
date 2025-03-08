'''
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
MoChaTo: Evaluation script for "Pattern in neutron scattering data and the 
topology of monochain molecules
-------------------------------------------------------------------------------
written by Jonas Soucek, 2025
at TU Dresden and Leibniz Institute of Polymer Research

This script it used to get and evaluate simulated data from an .hdf5 file. 

'''

import os
import sys
from os.path import exists
sys.path.append(os.path.dirname(__file__))
import library_MoChaTo as lib
import glob
import h5py


root_dir = r'C:\Users\jonas\OneDrive\TU Dresden\Physik-Studium, Bachelor\Abschlussarbeit'
search_crit = r'\**\*.hdf5'

system = 'windows'              # clearify operating systsem for file handling

filter_obj = 'swell_sqiso_key'
datagroups = [filter_obj, 'swell_sqiso']
eva_path = r'C:\Users\jonas\OneDrive\TU Dresden\Physik-Studium, Bachelor\Abschlussarbeit\script_evaluation'


for path in glob.glob(root_dir+search_crit, recursive=True):
    if not exists(eva_path + r'\results.txt'):
        os.makedirs(eva_path, exist_ok=True)

    with open(eva_path + r'\results.txt', 'w') as res_file:
        res_file.write(f'File: {path}\n')
        res_file.write('This is a results file of an .hdf5 file evaluation')
        res_file.write('containing a simulation of 500 monochain molecules\n')
        res_file.write('and thier resulting structural factor regarding')
        res_file.write('neutron scattering. It contains some information on\n')
        res_file.write('the operaing system and hardware as well as')
        res_file.write('signifficant quantities for further evaluation.\n\n')

        res_file.write(f'Number of PCA components: 2\n\n')

    with h5py.File(path, 'r') as file:
        file.visit(lambda x: lib.filter_func(name=x, file=file,\
            filter_obj=filter_obj, get_datasets=datagroups, eva_path=eva_path,\
            system=system, TestRun=False))