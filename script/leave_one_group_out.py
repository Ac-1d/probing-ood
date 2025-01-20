#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:20:16 2022

@author: kangming
"""
#%%

import pandas as pd
import pathlib  
from extra_funcs import leave_one_group_out, load_data,get_args,get_model

# parse arguments
dataset, target, group_label, group_value_list, modelname, force_rerun,summary_only = get_args()

# define model
'''
For a new model to be tested for this pipeline, the model should have a fit method and a predict method, namely
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

The fit method should reinitialize the model parameters every time a fit is called. 
This is because sometimes the model will be called multiple times and we want to make sure that the model is trained from scratch (rather than from a previous checkpoint) every time.

The model is defined in return_model() function in extra_funcs.py
'''

if summary_only: # if summary_only is True, then we don't really need to train the model. Just load the simplest model for consistency
    model = get_model(
            'xgb',
            dataset=dataset,
            target=target,
            group_label=group_label,
            group_value_list=group_value_list,
            )
    #load dataframe, and X,y
    df, X, y = load_data('llm',dataset,target)
else:
    if modelname == 'llm':
        model = None
    else:
        model = get_model(
                modelname,
                dataset=dataset,
                target=target,
                group_label=group_label,
                group_value_list=group_value_list,
                )
    #load dataframe, and X,y
    df, X, y = load_data(modelname,dataset,target)



#%%
''' 
Leave one attribute out 
'''
folder=f'leave_one_group_out/{target}'
pathlib.Path(f"./{folder}").mkdir(parents=True, exist_ok=True)


'''
csv_pred_prefix is a part of the csv filename csv_pred = f'{csv_pred_prefix}_{group_value_}.csv'
where group_value_ is the element in group_value_list, csv_pred is the csv file that contains the predictions of the structures in the test set
'''
csv_pred_prefix = f'{folder}/{dataset}_{modelname}_leave_one_{group_label}_out_pred'    
# the csv file that summarizes the results. it's a bit of a legacy part. non essentials.
csv_file = f'{folder}/{dataset}_{modelname}_leave_one_{group_label}_out.csv' 
scores = leave_one_group_out(group_label, group_value_list,
                            model, df, X, y,
                            csv_file = csv_file,csv_pred_prefix=csv_pred_prefix,
                            force_rerun=force_rerun, # if True, rerun even if the csv file already exists
                            summary_only=summary_only,
                            save_data_files=False, # if True, save the dataframes of the train and test sets, for debugging purposes
                                )

