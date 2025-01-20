#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming
"""
#%%
from extra_funcs import leave_one_group_out, load_data,get_args,set_model_output_dir,get_scores_from_pred,get_index
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import argparse
from matplotlib import pyplot as plt

#%%
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--dataset', type=str, default='jarvis22')
parser.add_argument('--target', type=str, default='e_form')
parser.add_argument('--group_label', type=str, default='elements')
parser.add_argument('--group_value', type=str, default='F')
parser.add_argument('--modelname', type=str,required=True)
args = parser.parse_args()

epochs = args.epochs
dataset = args.dataset
target = args.target
modelname = args.modelname
group_label = args.group_label
group_value = args.group_value

# #%%
# dataset = 'jarvis22'
# target = 'e_form'
# group_label = 'elements'
# group_value = 'Bi'
# modelname = 'xgb'
df, X, y = load_data(modelname,dataset,target)

#%%

val_ratio = 0.1
output_dir = f'learning_curve/{dataset}_{group_label}_{group_value}_{target}_{modelname}'
os.makedirs(output_dir,exist_ok=True)


#%%
index_trainval,index_test = get_index(df,group_label,group_value)

# Train/test split for (X,y)
X_trainval, y_trainval = X.loc[index_trainval], y.loc[index_trainval]
X_test, y_test = X.loc[index_test], y.loc[index_test]

# Train/val split for (X_trainval,y_trainval): val_ratio 
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=val_ratio, random_state=0)

mad_val = (y_val - y_val.mean()).abs().mean()
mad_test = (y_test - y_test.mean()).abs().mean()

#%%
if modelname == 'xgb':
    # skip if {output_dir}/evals_result.csv exists
    if os.path.exists(os.path.join(output_dir,'evals_result.csv')):
        print(f'Skipping the {modelname} training for {output_dir}')
    else:
        import xgboost as xgb
        model = xgb.XGBRegressor(
        n_estimators=1000, learning_rate=0.25,
        reg_lambda=0.01,reg_alpha=0.1,
        subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
        num_parallel_tree=4 ,tree_method='gpu_hist'
        )
        eval_set = [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
        model.fit(X_train, y_train, eval_metric=['mae',"rmse"], eval_set=eval_set, verbose=True)

        # New code to save eval metrics to CSV
        evals_result = model.evals_result()

        # Reshape the dictionary to the desired format
        reshaped_dict = {
            'train_mae': evals_result['validation_0']['mae'],
            'val_mae': evals_result['validation_1']['mae'],
            'test_mae': evals_result['validation_2']['mae'],
            'train_rmse': evals_result['validation_0']['rmse'],
            'val_rmse': evals_result['validation_1']['rmse'],
            'test_rmse': evals_result['validation_2']['rmse'],
        }

        evals_result = pd.DataFrame(reshaped_dict)
        evals_result.to_csv(os.path.join(output_dir, 'evals_result.csv'))


# elif modelname == 'rf':
#     from sklearn.ensemble import RandomForestRegressor
#     model = RandomForestRegressor(
#     n_estimators=100, max_features=1/3, n_jobs=-1, random_state=0
#     )

elif 'alignn' in modelname:
    from jarvis.db.jsonutils import loadjson
    from alignn.config import TrainingConfig
    from sklearnutils import AlignnLayerNorm
    config_filename = 'data/config.json'
    config = loadjson(config_filename)
    config = TrainingConfig(**config)

    config.epochs = int(modelname.replace('alignn',''))
    config.output_dir = output_dir

    model = AlignnLayerNorm(config) 
    # skip if {output_dir}/checkpoint_100.pt exists
    if os.path.exists(os.path.join(output_dir,'checkpoint_100.pt')) or os.path.exists(os.path.join(output_dir,'checkpoint_200.pt')):
        print(f'Skipping the {modelname} training for {output_dir}')
        
    else:
        model.fit(X_train, y_train, (X_val,y_val), (X_test, y_test))
    
elif modelname == 'gmp':
    try:
        from sksnn import skModel_CompressionSet2Set # get_gmp_features
        nsigmas, MCSH_order, width = 40, 4, 1.5
        batch_size = 256

        compressor_hidden_dim = [512,256,128,64,32]
        predictor_hidden_dims = [256,128,64,32,16]
        processing_steps = 9  # Number of processing steps in Set2Set
        num_layers = 5  # Number of layers in Set2Set

        if target == 'bandgap' or target == 'bulk_modulus':
            positive_output = True
        else:
            positive_output = False

        input_dim = nsigmas * (MCSH_order+1) + 1
        output_dir = output_dir

        model = skModel_CompressionSet2Set(
                input_dim, compressor_hidden_dim, predictor_hidden_dims, processing_steps, num_layers,positive_output,
                epochs,
                output_dir,
                val_ratio=val_ratio, batch_size=batch_size,
                )
        # skip if {output_dir}/checkpoint_100.pt exists
        if os.path.exists(os.path.join(output_dir,'log.dat')):
            print(f'Skipping the {modelname} training for {output_dir}')
            
        else:        
            model.fit(X_train, y_train, (X_val,y_val), (X_test, y_test))
        
    except Exception as e:
        print(f'Error in loading gmp model: {e}')

else:
    raise ValueError(f'Unknown model: {modelname}')


# %%
