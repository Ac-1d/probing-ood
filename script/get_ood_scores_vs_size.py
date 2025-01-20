#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:20:16 2022

@author: kangming
"""
#%%
import pandas as pd
import numpy as np
import os  
from extra_funcs import load_data,get_args,get_model,get_scores_from_pred, get_split
import argparse
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser(description='Leave one group out')
parser.add_argument('--dataset', type=str, 
                    default='jarvis22', 
                    help='Dataset name. Possible values: jarvis22 (76k entries), mp21 (146k entries), oqmd21 (1M entries)')
parser.add_argument('--target', type=str, 
                    default='e_form', 
                    help='Target property. Possible values: e_form, bandgap, bulk_modulus. Note that bulk_modulus is not available for oqmd21')
# parser.add_argument('--random_state', type=int, default=-1, help='The random state for the train test split.')
parser.add_argument('--modelname', type=str, required=True, help='The model name. ')
parser.add_argument('--group_label', type=str, help="""
        The grouping criterion. For example, Possible values:
            elements: leave one element out
            group: leave one group (column in the periodic table) out
            period: leave one period (row in the periodic table) out
            space_group_number: leave one space group out
            point_group: leave one point group out
            crystal_system: leave one crystal system out
            greater_than_nelements: leave structures with greater than n elements out
            le_nelements: leave structures with less than or equal to n elements out
            nelements: leave structures with n elements out
        """)
parser.add_argument('--group_value_list', type=str, help='''
        The list of values of the group_label, should be space-delimited. 
                    For example, if group_label is elements, then group_value_list should be a list of elements, e.g. ['H','He','Li',...].
                    If group_label is space_group_number, then group_value_list should be a list of space group numbers, e.g. [1,2,3,...].
        ''')
parser.add_argument('--train_ratio_list', 
                    default=None,
                    type=str, help='''
        The list of training ratios, should be space-delimited.
        ''')

# add the optional argument for the ood_train_size 
parser.add_argument('--ood_train_size', type=int, default=None, help='The training size for the OOD dataset.')

args = parser.parse_args()
modelname = args.modelname
dataset = args.dataset
target = args.target
# random_state = args.random_state
group_label = args.group_label
group_value_list = args.group_value_list.split()
train_ratio_list = args.train_ratio_list
ood_train_size = args.ood_train_size
if train_ratio_list is not None:
    train_ratio_list = train_ratio_list.split()


if train_ratio_list is None:
    train_ratio_list = [0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.33,1.0]
    # if dataset == 'oqmd21':
    #     train_ratio_list = '0.0001 0.0002 0.0004 0.0008 0.0016 0.0025 0.0036 0.0049 0.0064 0.01 0.02 0.04 0.09 0.16 0.25 0.36 0.49 0.64 0.81 1.00'.split()
    # else:
    #     train_ratio_list = '0.001 0.0016 0.0025 0.0036 0.0049 0.0064 0.01 0.02 0.04 0.09 0.16 0.25 0.36 0.49 0.64 0.81 1.00'.split()


    # if 'alignn' in modelname or 'gmp' in modelname:
    #     if dataset == 'oqmd21':
    #         train_ratio_list = [0.0001,0.0003,0.001,0.003,0.01,0.03,0.1,0.33,1.0]
    #     else:
    #         train_ratio_list = [0.001,0.003,0.007,0.01,0.03,0.09,0.25,0.5,1.0]

train_ratio_list = [float(x) for x in train_ratio_list]

#%% load dataframe, and X,y
df, X, y = load_data(modelname,dataset,target)

#%%

for group_value in group_value_list:
    index_pool, index_test = get_split(df, group_label,group_value)
    X_pool, y_pool = X.loc[index_pool], y.loc[index_pool]
    X_test_ood, y_test_ood = X.loc[index_test], y.loc[index_test]
    if ood_train_size is not None:
        X_test_ood_original, y_test_ood_original = X_test_ood.copy(), y_test_ood.copy()

    # split into training pool and in-distribution test
    X_trainpool, X_test_id, y_trainpool, y_test_id = train_test_split(X_pool, y_pool, train_size=0.8, random_state=0)

    for train_ratio in train_ratio_list:

        if modelname in ['xgb','rf']:
            if train_ratio ==1:
                random_state_list = np.arange(2)
            elif train_ratio >0.3:
                random_state_list = np.arange(4)
            elif train_ratio >0.16:
                random_state_list = np.arange(5)
            elif train_ratio >0.1:
                random_state_list = np.arange(6)
            # elif train_ratio >=0.01:
            #     random_state_list = np.arange(10)
            else:
                random_state_list = np.arange(10)
        elif 'alignn' in modelname or 'gmp' in modelname:
            if dataset == 'oqmd21': 
                if train_ratio >=0.5:
                    random_state_list=[0]
                elif train_ratio > 0.1:
                    random_state_list=[0,1]
                elif train_ratio >0.01:
                    random_state_list=[0,1,2,3]
                elif train_ratio >0.001:
                    random_state_list=[0,1,2,3,4,5,6,7]
                else:
                    random_state_list=[0,1,2,3,4,5,6,7,8,9]

            else:
                if train_ratio ==1:
                    random_state_list=[0]
                elif train_ratio >0.4:
                    random_state_list=[0,1]
                elif train_ratio >0.3:
                    random_state_list=[0,1,2]
                elif train_ratio >0.1:
                    random_state_list=[0,1,2,3]
                elif train_ratio >=0.01:
                    random_state_list=[0,1,2,3,4]
                else:
                    random_state_list=[0,1,2,3,4,5,6,7,8,9]

        for random_state in random_state_list:
            if ood_train_size is not None:
                output_dir = f'output_id/{modelname}/{dataset}/{target}/{group_label}_{group_value}_{train_ratio}/{random_state}_ood_train_size_{ood_train_size}'
            else:
                output_dir = f'output_id/{modelname}/{dataset}/{target}/{group_label}_{group_value}_{train_ratio}/{random_state}'
            csv_id = f'{output_dir}/y_pred_id.csv'
            csv_ood = f'{output_dir}/y_pred_ood.csv'

            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            if not os.path.exists(csv_ood):
                # define model
                model = get_model(modelname,output_dir=output_dir,random_state=random_state)

                # use a subset of the training pool
                if train_ratio == 1:
                    X_train, y_train = X_trainpool, y_trainpool
                else:
                    X_train, _, y_train, _ = train_test_split(X_trainpool, y_trainpool, train_size=train_ratio, random_state=random_state)        

                if ood_train_size > 0:
                    # get a subset of the OOD test set for training
                    X_train_ood, _, y_train_ood, _ = train_test_split(X_test_ood_original, y_test_ood_original, 
                                                                      train_size=ood_train_size, random_state=random_state)
                    X_train = pd.concat([X_train, X_train_ood])
                    y_train = pd.concat([y_train, y_train_ood])
                    # remove the training data from the OOD test set
                    X_test_ood = X_test_ood_original.drop(X_train_ood.index)
                    y_test_ood = y_test_ood_original.drop(y_train_ood.index)

                # print the size of X_train, X_test_id, X_test_ood
                print(f'X_train: {X_train.shape}, X_test_id: {X_test_id.shape}, X_test_ood: {X_test_ood.shape}')

                # train
                model.fit(X_train, y_train)
                # predict
                y_pred_id = model.predict(X_test_id)
                y_pred_ood = model.predict(X_test_ood)
                # save
                df_y_id = pd.DataFrame({'y_pred':y_pred_id,'y_test':y_test_id},index=y_test_id.index)
                df_y_ood = pd.DataFrame({'y_pred':y_pred_ood,'y_test':y_test_ood},index=y_test_ood.index)
                df_y_id.to_csv(csv_id)
                df_y_ood.to_csv(csv_ood)
                print(f'{csv_id} saved.')
            else:                
                print(f'csv {csv_ood} exists. Reading from file...')
                df_y_id = pd.read_csv(csv_id,index_col=0)
                df_y_ood = pd.read_csv(csv_ood,index_col=0)
                y_pred_id = df_y_id['y_pred']
                y_test_id = df_y_id['y_test']
                y_pred_ood = df_y_ood['y_pred']
                y_test_ood = df_y_ood['y_test']


            # get ID scores
            mad, std, maes, rmse, r2, pearson_r, pearson_p_value, spearman_r, spearman_p_value, kendall_r, kendall_p_value = get_scores_from_pred(y_test_id,y_pred_id)
            # print mae and r2
            print(f'ID mae: {maes:.3f}, r2: {r2:.3f}')
            # get OOD scores
            mad, std, maes, rmse, r2, pearson_r, pearson_p_value, spearman_r, spearman_p_value, kendall_r, kendall_p_value = get_scores_from_pred(y_test_ood,y_pred_ood)
            # print mae and r2
            print(f'OOD mae: {maes:.3f}, r2: {r2:.3f}')
            print('')


