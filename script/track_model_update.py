#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming
"""
#%%
from extra_funcs import separate_contrib, load_data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.model_selection import train_test_split
from myfunc import get_col2drop
import pickle
import os
import matplotlib.colors as mcolors

dataset = 'mp21'
target = 'e_form'
modelname = 'xgb'

#%%

df, X, y = load_data(modelname,dataset,target)

#%%
# remove columns with zero variance
X = X.loc[:,X.std()!=0]
col2keep, col2drop = get_col2drop(X,cutoff=0.7)
X = X.loc[:,col2keep]


#%%

def get_split(X,y, group_label,group_value,train_size=0.8):
    if group_label == 'point_group':
        index_train = df[df['point_group']!=group_value].index
        index_test = df[df['point_group']==group_value].index
    elif group_label == 'elements':
        index_train = df[df['elements'].apply(lambda x: group_value not in x)].index
        index_test = df[df['elements'].apply(lambda x: group_value in x)].index

    X_train, y_train = X.loc[index_train], y.loc[index_train]
    X_test, y_test = X.loc[index_test], y.loc[index_test]
    # split the test set into two parts
    if train_size == 1:
        X_test2, y_test2 = None, None
        # print dataset size
        print('X_train: ',X_train.shape)
        print('X_test: ',X_test.shape)
    else:
        X_test, X_test2, y_test, y_test2 = train_test_split(
            X_test,y_test,
            random_state=0,train_size=train_size
            )
        # print dataset size
        print('X_train: ',X_train.shape)
        print('X_test: ',X_test.shape)
        print('X_test2: ',X_test2.shape)
    return X_train, y_train, X_test, y_test, X_test2, y_test2



group_label = 'elements'
# group_label = 'point_group'

if group_label == 'point_group':
    group_value_list = list(set(df[group_label].tolist()))
elif group_label == 'elements':
    # group_value_list = ['F','H','O','N','C','Cl','S','P','B',
    #                     'Si','Se','Te','I','Br','Cr','Mn','Fe',
    #                     'Co','Ni','Cu','Zn','Ga','Al','Li',
    #                     'Na','Be','K','Bi','Hg','Rb','Ba','Pt','Cs','Pd']
    group_value_list = ['H','F','O','P','I','Li', 'Pt', 'Bi','Fe','Rb']

struct_compo='struct'


pkl_shap = f'shap_values/{struct_compo}/{dataset}_{target}_{modelname}_{group_label}_shap.pkl'
if os.path.exists(pkl_shap):
    with open(pkl_shap,'rb') as f:
        shap_compo, shap_struc = pickle.load(f)
else:
    shap_compo, shap_struc = {},{}


train_size = 1

for group_value in group_value_list:
    print('')
    print('group_value: ',group_value)
    if group_value in shap_compo.keys():
        print('Already calculated')
        continue

    X_train, y_train, X_test, y_test, X_test2, y_test2 = get_split(
        X,y,group_label,group_value,train_size=train_size
        )
    if len(y_test) == 0:
        continue
    shap_compo[group_value], shap_struc[group_value] = separate_contrib(
        X_train, y_train, X_test, y_test, X_test2, y_test2,struct_compo=struct_compo
    )
    
# save shap values
with open(pkl_shap,'wb') as f:
    pickle.dump([shap_compo, shap_struc],f)


#%%

import seaborn as sns
data = []
# ['F','H','O','N','C','Cl','S','P','B','Si','Se','Te','I','Br','Cr','Mn','Fe','Co','Ni','Cu','Zn','Ga']
# group_value_list = ['-43m', 'm-3m', 'm','4mm','-6m2','-1'] 
# group_value_list =  ['H','F','O','P','I','Li', 'Pt', 'Bi','Fe','Rb']
group_value_list =  ['H','F','O','Cl']

for group_value in group_value_list:
    shap_compo_sum = shap_compo[group_value].sum(axis=1)
    shap_struc_sum = shap_struc[group_value].sum(axis=1)
    for value, type_ in zip(shap_compo_sum, np.repeat('Compositional', len(shap_compo_sum))):
        data.append([group_value, value, type_])
    for value, type_ in zip(shap_struc_sum, np.repeat('Structural', len(shap_struc_sum))):
        data.append([group_value, value, type_])

df_vio = pd.DataFrame(data, columns=['group_value', 'value', 'type'])
    
fig, ax = plt.subplots(figsize=(4, 3)) # figsize=(6.5, 3)
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
sns.violinplot(x='group_value', y='value', hue='type', 
               data=df_vio, split=True, ax=ax, inner=None, 
               linewidth=1, palette=['#1f77b4', '#ff7f0e'], 
            #    density_norm='width',
               )
if group_label == 'elements':
    ax.set_yticks(np.arange(-1.5,1.5,0.5))
    ax.set_yticklabels(np.arange(-1.5,1.5,0.5))
    ax.set_ylim(-1.5,0.75)
    ax.legend(loc=(0.01,0.84),fontsize=8.5)
elif group_label == 'point_group':
    ax.set_ylim(-0.4,0.2)
    ax.legend(loc='lower right')
ax.set_xlabel('')
ax.set_ylabel('SHAP contribution (eV/atom)')

fig.savefig('figs/paper/shap_violin.png', bbox_inches='tight', dpi=300)

# %%
