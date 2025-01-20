#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming
"""

#%%
import os
import numpy as np
import pandas as pd
import shutil 

# import modin.pandas as pd
# import ray
# ray.init()

from myfunc import dictAtoms2objStructure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer   
from pymatgen.core.structure import Structure
import ast
# import swifter

cwd = os.getcwd()


#%%

def add_attribute(df):
    # from tqdm import tqdm
    # tqdm.pandas()
    '''
    Add attributes to the dataframe
    '''
    # convert elements to list of strings
    if 'elements' not in df.columns:
        df['elements'] = df['structure'].apply(
            lambda x: x.composition.elements
            ).apply(lambda x: [str(e) for e in x])
    
    if 'space_group_number' not in df.columns:
        df['space_group_number'] = df['structure'].apply(
            lambda x: x.get_space_group_info()[1]
            )
        
    if 'point_group' not in df.columns:
        df['point_group'] = df['structure'].apply(
            lambda x: SpacegroupAnalyzer(x).get_point_group_symbol()
            )
        
    if 'crystal_system' not in df.columns:
        df['crystal_system'] = df['structure'].apply(
                    lambda x: SpacegroupAnalyzer(x).get_crystal_system()
                    )
    
    return df



col_targets = ['e_form','bandgap','bulk_modulus']
col_attributes = ['elements','space_group_number','point_group','crystal_system']
# read matminer_features from the file
with open('matminer_feature_labels.txt','r') as f:
    matminer_features = f.read().splitlines()
col_features = ['precomputed_graphs'] + matminer_features
col_features_nograph = matminer_features
cols = col_targets + col_attributes + col_features
cols_nograph = col_targets + col_attributes + col_features_nograph


datadir = '/home/kangming/Gdrive/UofT/Coding/auto_alloys/paper-dataset-distillation/organized/data'


#%%

'''
Jarvis dataset
'''

df = pd.read_pickle(datadir+'/jarvis22/dat_featurized.pkl')
print('jarvis dataset loaded')

# add structure column
df2 = pd.read_json(datadir+'/jarvis22/jdft_3d-12-12-2022.json').set_index('jid')
df['structure'] = df2['atoms'].apply(dictAtoms2objStructure)

print('starting to add attributes')
df = add_attribute(df)

print('saving to pickle')
df[cols].to_pickle('data/jarvis22/dat_featurized.pkl')
df[cols_nograph].to_pickle('data/jarvis22/dat_featurized_matminer.pkl')

#%%
structure = pd.read_json(datadir+'/jarvis22/jdft_3d-12-12-2022.json').set_index('jid')['atoms'].apply(dictAtoms2objStructure)
structure.to_pickle('data/jarvis22/jarvis22_pmg_structure.pkl')
#make a folder to store cif files
os.mkdir('data/jarvis22/jarvis22_cif')
for jid, s in structure.items():
    s.to('cif','data/jarvis22/jarvis22_cif/'+jid+'.cif')
# zip the folder
shutil.make_archive('data/jarvis22/jarvis22_cif', 'zip', 'data/jarvis22/jarvis22_cif')
# remove the folder
shutil.rmtree('data/jarvis22/jarvis22_cif')





#%%
'''
mp dataset
'''
df = pd.read_pickle(datadir+'/mp/dat_featurized.pkl')
print('mp dataset loaded')

# metadata
df['structure'] = pd.read_pickle(datadir+'/mp/dat.pkl')['structure']

# print('starting to add attributes')
# df = add_attribute(df)

# print('saving to pickle')
# df[cols].to_pickle('data/mp21/dat_featurized.pkl')
# df[cols_nograph].to_pickle('data/mp21/dat_featurized_matminer.pkl')


structure = pd.read_pickle(datadir+'/mp/dat.pkl')['structure']
structure.to_pickle('data/mp21/mp21_pmg_structure.pkl')
#make a folder to store cif files
os.mkdir('data/mp21/mp21_cif')
for mp_id, s in structure.items():
    s.to('cif','data/mp21/mp21_cif/'+mp_id+'.cif')
# zip the folder
shutil.make_archive('data/mp21/mp21_cif', 'zip', 'data/mp21/mp21_cif')
# remove the folder
shutil.rmtree('data/mp21/mp21_cif')


#%%




#%%




#%%
'''
oqmd dataset
'''
# df2 = pd.read_pickle(datadir+'/oqmd/dat_featurized0.pkl')

# structure = pd.DataFrame()

# structure['structure'] = df2['structure_as_dict'].apply(
#     lambda x: Structure.from_dict(ast.literal_eval(x))
# )

# print('starting to add attributes')
# s = add_attribute(structure)

# s.to_pickle('data/oqmd21/structure.pkl')
# s[col_attributes].to_pickle('data/oqmd21/col_attributes.pkl')


''' Close the terminal and reload '''

s  = pd.read_pickle('data/oqmd21/col_attributes.pkl')

df2 = pd.read_pickle(datadir+'/oqmd/dat_featurized.pkl')
print('oqmd dataset loaded')

df = pd.concat([df2,s],axis=1)

print('saving to pickle')

cols.remove('bulk_modulus')
cols_nograph.remove('bulk_modulus')

df[cols_nograph].to_pickle('data/oqmd21/dat_featurized_matminer.pkl')
df[cols].to_pickle('data/oqmd21/dat_featurized.pkl')

#%%

structure = df2['structure_as_dict'].apply(
    lambda x: Structure.from_dict(ast.literal_eval(x))
)
structure.to_pickle('data/oqmd21/oqmd21_pmg_structure.pkl')
#make a folder to store cif files
os.mkdir('data/oqmd21/oqmd21_cif')
for i, s in structure.items():
    s.to('cif','data/oqmd21/oqmd21_cif/'+str(i)+'.cif')
# zip the folder
shutil.make_archive('data/oqmd21/oqmd21_cif', 'zip', 'data/oqmd21/oqmd21_cif')
# remove the folder
shutil.rmtree('data/oqmd21/oqmd21_cif')




#%%





#%%
'''
https://pages.nist.gov/jarvis_leaderboard/AI/SinglePropertyPrediction/

worth considering qe_tb and hmof (different properties)

qm9 as an example for molecular data


check different properties to check the extrapolability

'''


#%%

# later for ALIGNN: no need to do entries fewer than 1000 test entries (for saving compute cost), and for space group
# how does training with different epoches change the extrapolation (vs. interpolation) performance?


