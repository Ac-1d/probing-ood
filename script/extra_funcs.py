#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming
"""
import pandas as pd
import os
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn import metrics
import time
import numpy as np

# import the pymatgen element class
from pymatgen.core.periodic_table import Element
import argparse
try: 
    from llmprop_ood_utils import train_and_predict_main
except:
    pass

def get_args():
    parser = argparse.ArgumentParser(description='Leave one group out')
    parser.add_argument('--dataset', type=str, help='Dataset name. Possible values: jarvis22 (76k entries), mp21 (146k entries), oqmd21 (1M entries)')
    parser.add_argument('--target', type=str, help='Target property. Possible values: e_form, bandgap, bulk_modulus. Note that bulk_modulus is not available for oqmd21')
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
    parser.add_argument('--modelname', type=str, help="""
                        Model name. Used as a part of the csv filename to save the results.
                        Model name also determines the type of data/features used.                    
                        """)
    parser.add_argument('--force_rerun', action=argparse.BooleanOptionalAction, help='If True, rerun even if the csv file already exists')
    parser.add_argument('--summary_only', action=argparse.BooleanOptionalAction, help='If True, only get the summary of the results, and skip the ML training ')
    args = parser.parse_args()
    dataset = args.dataset
    target = args.target
    group_label = args.group_label
    group_value_list = args.group_value_list.split()
    modelname = args.modelname 
    force_rerun = args.force_rerun
    summary_only = args.summary_only
    if force_rerun is True:
        print('force_rerun is True. Will rerun even if the csv file already exists')


    # Check if arguments are valid
    avail_datasets = ['jarvis22','mp21','oqmd21']
    avail_targets = ['e_form','bandgap','bulk_modulus']
    avail_modelnames = ['xgb','rf','alignn','llm','gmp']
    avail_group_labels = ['elements','group','period','space_group_number','point_group','crystal_system',
                        'greater_than_nelements','le_nelements','nelements','prototype']

    if dataset not in avail_datasets:
        raise ValueError(f'dataset must be one of {avail_datasets}. Got {dataset}')
    if target not in avail_targets:
        raise ValueError(f'target must be one of {avail_targets}. Got {target}')
    # if modelname not in avail_modelnames  and ('alignn' not in modelname):
    #     raise ValueError(f'modelname must be one of {avail_modelnames}. Got {modelname}')
    # if target == 'bulk_modulus' and 'oqmd' in dataset:
    #     raise ValueError(f'bulk_modulus is not available for oqmd')

    # check that the group_label is valid
    if group_label not in avail_group_labels:
        raise ValueError(f'group_label must be one of {avail_group_labels}. Got {group_label}')

    # convert group_value_list to int if group_label is one of the following
    if group_label in ['group','period','space_group_number','greater_than_nelements',
                        'le_nelements','nelements']:
        group_value_list = [int(i) for i in group_value_list]

    return dataset, target, group_label, group_value_list, modelname, force_rerun, summary_only


def get_model(
        modelname,
        dataset=None,
        target=None,
        group_label=None,
        group_value_list=None,
        output_dir=None,
        random_state=0,
        ):
    '''
    For a new model to be tested for this pipeline, the model should have a fit method and a predict method, namely
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)

    The fit method should reinitialize the model parameters every time a fit is called. 
    This is because sometimes the model will be called multiple times and we want to make sure that the model is trained from scratch (rather than from a previous checkpoint) every time.

    The model is defined in get_model() function in extra_funcs.py
    '''

    if output_dir is None:
        output_dir = set_model_output_dir(group_label,group_value_list,modelname,dataset,target)

    if modelname == 'xgb':
        import xgboost as xgb
        # check if xgb version is below 2.0
        if int(xgb.__version__.split('.')[0]) < 2:
            model = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.25,
            reg_lambda=0.01,reg_alpha=0.1,
            subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
            num_parallel_tree=4 ,
            tree_method='gpu_hist',
            random_state=random_state,
            )
        else:
            model = xgb.XGBRegressor(
            n_estimators=1000, learning_rate=0.25,
            reg_lambda=0.01,reg_alpha=0.1,
            subsample=0.85,colsample_bytree=0.3,colsample_bylevel=0.5,
            num_parallel_tree=4 ,
            tree_method = "hist", device = "cuda",
            random_state=random_state,
            )

    elif modelname == 'rf':
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(
        n_estimators=100, max_features=1/3, n_jobs=-1, random_state=random_state
        )


    elif 'alignn' in modelname:
        from jarvis.db.jsonutils import loadjson
        from alignn.config import TrainingConfig
        from sklearnutils import AlignnLayerNorm
        config_filename = 'data/config.json'
        config = loadjson(config_filename)
        config = TrainingConfig(**config)

        config.epochs = int(modelname.replace('alignn',''))
        config.output_dir = output_dir
        config.target = target
        model = AlignnLayerNorm(config)         
        
    elif 'gmp' in modelname:
            
        nsigmas, MCSH_order, width = 40, 4, 1.5
        batch_size = 256
        val_ratio = 0.1
        epochs = 200
        input_dim = nsigmas * (MCSH_order+1) + 1

        if 'e_form' not in modelname:
            from sksnn import skModel_CompressionSet2Set # get_gmp_features
            compressor_hidden_dim = [512,256,128,64,32]
            predictor_hidden_dim = [256,128,64,32,16]
            processing_steps = 9  # Number of processing steps in Set2Set
            num_layers = 5  # Number of layers in Set2Set

            if target == 'bandgap' or target == 'bulk_modulus':
                positive_output = True
            else:
                positive_output = False

            model = skModel_CompressionSet2Set(
                    input_dim, compressor_hidden_dim, predictor_hidden_dim, 
                    processing_steps, num_layers,positive_output,
                    epochs,
                    output_dir,
                    val_ratio=val_ratio, batch_size=batch_size,
                    )        

        else:
            from sksnn import skModel_SimpleSum            
            hidden_dims = [256,128,64]
            model = skModel_SimpleSum(
                input_dim, hidden_dims, 
                epochs,
                output_dir,
                val_ratio=val_ratio, batch_size=batch_size,
                )



    else:
        raise ValueError(f'Unknown model: {modelname}')
    
    return model



def get_index(df,group_label,group_value):
    if group_label == 'elements':
        index_test = df[df[group_label].apply(lambda x: group_value in x)].index.tolist()
        index_trainval = df[df[group_label].apply(lambda x: group_value not in x)].index.tolist()
    else: 
        index_test = df[df[group_label]==group_value].index.tolist()
        index_trainval = df[df[group_label]!=group_value].index.tolist()
    return index_trainval,index_test


def set_model_output_dir(group_label,group_value_list,modelname,dataset,target):
    if group_value_list is None:
        return None

    if len(group_value_list) == 1 :
        group_value_ = group_value_list[0]
        if isinstance(group_value_,str):
            group_value_ = group_value_.replace('/','_')
        output_dir = f'output_{modelname}/{dataset}_{target}_{group_label}_{group_value_}'
    else:
        output_dir = f'output_{modelname}/{dataset}_{target}_{group_label}_many'
    return output_dir


def get_split(df, group_label,group_value):
    if group_label in ['space_group_number','point_group','crystal_system',
                       ]:
        index_train = df[df[group_label].astype(str)!=group_value].index
        index_test = df[df[group_label].astype(str)==group_value].index
    elif group_label in ['elements','period','group',]:
        index_train = df[df[group_label].apply(lambda x: group_value not in x)].index
        index_test = df[df[group_label].apply(lambda x: group_value in x)].index
    elif group_label == 'greater_than_nelements':
        index_train = df[df['nelements']<=int(group_value)].index
        index_test = df[df['nelements']>int(group_value)].index        
    else:
        raise NotImplementedError
    return index_train, index_test

def load_data(modelname,dataset,target):
    def load_matminer_data(dataset,target):
        # read matminer feature names from the file
        with open('data/matminer_feature_labels.txt','r') as f:
            matminer_features = f.read().splitlines()
        
        # df = pd.read_json(f'data/{dataset}/dat_featurized_matminer.json')
        df = pd.read_pickle(f'data/{dataset}/dat_featurized_matminer.pkl')   
        # drop entries whose e_form is larger than 5 eV/atom
        df = df[df['e_form'] < 5]
        # drop entries whose target or features are NaN
        df = df.dropna(subset=[target]+matminer_features)   
        df = add_elemental_attributes(df)
        X = df[matminer_features].astype(float)
        y = df[target]
        return df, X, y

    def load_alignn_data(dataset,target):
        df = pd.read_pickle(f'data/{dataset}/dat_featurized.pkl')    
        # drop entries whose e_form is larger than 5 eV/atom
        df = df[df['e_form'] < 5]
        # drop entries whose target or features are NaN
        df = df.dropna(subset=[target])
        df = add_elemental_attributes(df)
        X = df['precomputed_graphs']
        y = df[target]
        return df, X, y

    def load_llm_data(dataset,target):
        try:
            df = pd.read_pickle(f'data/{dataset}/dat_featurized_llm.pkl')
        except:
            df = pd.read_json(f'data/{dataset}/dat_featurized_llm.json')
        # drop entries whose e_form is larger than 5 eV/atom
        df = df[df['e_form'] < 5]
        # drop entries whose target or features are NaN
        df = df.dropna(subset=[target,'description'])  
        X = df['description']
        y = df[target]
        return df, X, y
    
    def load_gmp_data(dataset,target):
        df = pd.read_pickle(f'data/{dataset}/dat_featurized_gmp.pkl')
        # drop entries whose e_form is larger than 5 eV/atom
        df = df[df['e_form'] < 5]
        # drop entries whose target or features are NaN
        df = df.dropna(subset=[target,'gmp_features'])
        df = add_elemental_attributes(df)
        X = df['gmp_features']
        y = df[target]
        return df, X, y

    if modelname in ['xgb','rf','lf']:
        df, X, y = load_matminer_data(dataset,target)
    elif 'alignn' in modelname:
        df, X, y = load_alignn_data(dataset,target)
    elif modelname == 'llm':
        df, X, y = load_llm_data(dataset,target)
    elif 'gmp' in modelname:
        df, X, y = load_gmp_data(dataset,target)
    else:
        raise ValueError(f'Data loading for {modelname} is not implemented.')
    return df, X, y


def add_NN_related_attributes(df):
    NN_related = ['minimum CN_VoronoiNN',
                  'maximum CN_VoronoiNN',
                  'range CN_VoronoiNN',
                  'mean CN_VoronoiNN',
                  'avg_dev CN_VoronoiNN']
    # # calculate the pearson correlation coefficient between NN_related attributes.
    # corr = df[NN_related].corr()
    small_avg_dev = (df['avg_dev CN_VoronoiNN']<2)
    large_avg_dev = (df['avg_dev CN_VoronoiNN']>=2)

    # g1: can leave out large_avg_dev_CN_VoronoiNN

    # g2: intersection of small_avg_dev_CN_VoronoiNN and mean_CN_VoronoiNN<12
    small_mean = (df['mean CN_VoronoiNN']<7)
    (small_mean & small_avg_dev).sum()

def add_elemental_attributes(df):    
    elements = df['elements'].apply(lambda x: [Element(e) for e in x])
    nelements = elements.apply(lambda x: len(set(x)))
    # get the group number of each element
    group = elements.apply(lambda x: [e.group for e in x])
    # get the period number of each element
    period = elements.apply(lambda x: [e.row for e in x])

    # # get the atomic number of each element
    # atomic_number = elements.apply(lambda x: [e.Z for e in x])
    # # get the atomic mass of each element
    # atomic_mass = elements.apply(lambda x: [e.atomic_mass for e in x])
    # # get the atomic radius of each element
    # atomic_radius = elements.apply(lambda x: [e.atomic_radius for e in x])
    # # get the electronegativity of each element
    # electronegativity = elements.apply(lambda x: [e.X for e in x])
    # # get the number of valence electrons of each element
    # nvalence = elements.apply(lambda x: [e.NValence for e in x])

    df['group'] = group
    df['period'] = period
    df['nelements'] = nelements
    return df


def get_scores_from_pred(y_test,y_pred):
    mad = (y_test - y_test.mean()).abs().mean()
    std = y_test.std()

    maes = metrics.mean_absolute_error(y_test,y_pred)
    rmse = metrics.mean_squared_error(y_test,y_pred,squared=False)
    r2 = metrics.r2_score(y_test,y_pred)

    # calculate correlation between y_test and y_pred
    pearson_r, pearson_p_value = pearsonr(y_test,y_pred)
    spearman_r, spearman_p_value = spearmanr(y_test,y_pred)
    kendall_r, kendall_p_value = kendalltau(y_test,y_pred)
    return mad, std, maes, rmse, r2, pearson_r, pearson_p_value, spearman_r, spearman_p_value, kendall_r, kendall_p_value

def parity_plot(y_test, y_pred):
    import matplotlib.pyplot as plt
    xmin = min(y_test.min(), y_pred.min())
    xmax = max(y_test.max(), y_pred.max())
    ymin, ymax = xmin, xmax

    plt.scatter(y_test, y_pred,alpha=0.5)
    plt.plot([xmin, xmax], [ymin, ymax], 'k--')
    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])
    plt.xlabel('DFT')
    plt.ylabel('ML')
    plt.show()

def print_scores(y_test, y_pred,plot=True):
    mad, std, maes, rmse, r2, pearson_r, pearson_p_value, spearman_r, spearman_p_value, kendall_r, kendall_p_value = get_scores_from_pred(y_test, y_pred)
    print(f'MAD: {mad:.3f}')
    print(f'MAE: {maes:.3f}')
    print(f'R2: {r2:.3f}')
    print(f'Pearson r: {pearson_r:.3f}')
    if plot:
        parity_plot(y_test, y_pred)

def leave_one_group_out(
        group_label, group_value_list, model, 
        df, X, y,
        csv_file = None,csv_pred_prefix=None,
        nmin = 200, # skip if n_entries < nmin, n_entries is the number of entries to be removed from the training set, and used as the test set
        cv_for_train = False, # default: use the first 80% of the training set as the training set, no 5 fold cv
        invert = False, # all filenames should be "inverted"
        use_pruned = False, ids_left = None,
        baseline_csv = None,
        base_model = 'rf',
        force_rerun = False,
        summary_only=False,
        save_data_files=False,
        ):
    scores = []
    scores_train = []
    scores_test = []
    chemical_group_label = ['elements','group','period']

    if csv_file is not None:
        # check if csv_file exists
        if os.path.exists(csv_file):
            print(f'{csv_file} found, updating it')
        else:
            print(f'{csv_file} not found, calculating scores from scratch')

    for group_value in group_value_list:    

        # Define the group to be left out
        if group_label in chemical_group_label:
            # used for defining a group of structures that contain multiple elements at the same time
            if isinstance(group_value, list) or isinstance(group_value, tuple):                
                group = (df[group_label].apply(
                    lambda x: set(group_value) <= set(x)
                    ))
            elif isinstance(group_value, str) or isinstance(group_value, Element) or isinstance(group_value, int):              
                group = (df[group_label].apply(
                    lambda x: group_value in x
                    ))
                
        elif group_label == 'greater_than_nelements': 
            # greater than nelements. 
            # In this case, group_value should take 2 (predict on ternary or more), 3 (predict on quaternary or more), 4 (predict on quinary or more), etc.
            group = (df['nelements'] > group_value)
        
        elif group_label == 'le_nelements': 
            # less or equal to nelements
            # In this case, group_value should take 2 (predict on binary or less), 3 (predict on ternary or less), 4 (predict on quaternary or less), etc.
            group = (df['nelements'] <= group_value) 
            
        elif group_label == 'prototype':
            group = df['description'].apply(lambda x: x is not None and group_value in x)

        else:
            group = (df[group_label] == group_value)


        if csv_pred_prefix is not None:
            if isinstance(group_value, list) or isinstance(group_value, tuple):            
                group_value_ = '_'.join(group_value)
            else:
                group_value_ = str(group_value).replace('/','') # remove '/' in the group_value, this is used when group_value is s point group which could contain '/' in its name

            csv_pred = f'{csv_pred_prefix}_{group_value_}.csv'

            csv_train_cv = csv_pred.replace('.csv','_train_cv.csv')
        else:
            csv_pred = None
            csv_train_cv = None
        
        X_train,y_train,X_test,y_test = X[~group],y[~group],X[group],y[group]       


        # if use_pruned:
        #     if ids_left is None:
        #         pass # to raise error
        #     X_train = X_train[X_train.index.isin(ids_left)]
        #     y_train = y_train[y_train.index.isin(ids_left)]

        n_entries = group[group].shape[0]
        ratio_entries = n_entries / group.shape[0]


        columns = ['n_entries', 'ratio_entries', 'mad', 'std', 'maes','rmse','r2',
                     'pearson_r', 'pearson_p_value', 'spearman_r', 'spearman_p_value', 'kendall_r', 'kendall_p_value']

        # skip if n_entries < nmin
        if X_train.shape[0]<nmin or X_test.shape[0]<nmin:
            (mad, std, maes, rmse, r2,
            pearson_r, pearson_p_value, spearman_r, spearman_p_value, kendall_r, kendall_p_value) = [None] * (len(columns)-2)

            (mad_train, std_train, maes_train, rmse_train, r2_train,
            pearson_r_train, pearson_p_value_train, spearman_r_train, spearman_p_value_train, kendall_r_train, kendall_p_value_train) = [None] * (len(columns)-2
             )
            
            if X_train.shape[0]<nmin:
                print(f'{group_label}={group_value}: {X_train.shape[0]} entries in the training set < {nmin}. Skipping...')
            if X_test.shape[0]<nmin:
                print(f'{group_label}={group_value}: {X_test.shape[0]} entries in the test set < {nmin}. Skipping...')

        else:
            # # Get the cv estimates on the training set
            # start_time = time.time()
            # if csv_train_cv is not None and os.path.exists(csv_train_cv):
            #     y_train_df = pd.read_csv(csv_train_cv,index_col=0)
            #     y_train_test = y_train_df['y_test']
            #     y_train_pred = y_train_df['y_pred']
            # else:
            #     # Get the scores on the training set
            #     # if cv_for_train, use 5 fold cv                
            #     if cv_for_train:
            #         y_train_test = y_train
            #         y_train_pred = cross_val_predict(model, X_train, y_train, cv=5)
            #     # 
            #     else:
            #         # use the first 80% of the training set as the training set
            #         n_train = int(X_train.shape[0]*0.8)
            #         X_train_train, X_train_test, y_train_train, y_train_test = X_train.iloc[:n_train], X_train.iloc[n_train:], y_train.iloc[:n_train], y_train.iloc[n_train:]
            #         model.fit(X_train_train, y_train_train)
            #         y_train_pred = model.predict(X_train_test)

            #     y_train_df = pd.DataFrame({'y_test': y_train_test, 'y_pred': y_train_pred}, index=y_train_test.index)                
            #     y_train_df.to_csv(csv_train_cv)

            # mad_train, std_train, maes_train, rmse_train, r2_train, pearson_r_train, pearson_p_value_train, spearman_r_train, spearman_p_value_train, kendall_r_train, kendall_p_value_train = get_scores_from_pred(y_train_test,y_train_pred)
            # time_elapsed = time.time() - start_time
            # print('')
            # print(f'{group_label}={str(group_value)}: {X_train.shape[0]}, {mad_train:.3f}, {std_train:.3f}')
            # print(f'Train_test scores: MAE={maes_train:.3f}, RMSE={rmse_train:.3f}, R2={r2_train:.3f}')
            # print(f'Pearson: r={pearson_r_train:.3f}, p={pearson_p_value_train:.3f}')
            # print(f'Spearman: r={spearman_r_train:.3f}, p={spearman_p_value_train:.3f}')
            # print(f'Kendall: r={kendall_r_train:.3f}, p={kendall_p_value_train:.3f}')
            # print(f'Time elapsed: {time_elapsed:.3f} s')
            # print('',flush=True)


            # get the previously calculated cv estimate of the whole dataset
            if baseline_csv is not None:
                y_df = pd.read_csv(baseline_csv,index_col=0)
                y_pred = y_df.loc[y_test.index,'y_pred']
                mad, std, maes_test, rmse_test, r2_test, pearson_r_test, pearson_p_value_test, spearman_r_test, spearman_p_value_test, kendall_r_test, kendall_p_value_test = get_scores_from_pred(y_test,y_pred)


            # Get the scores on the OOD test set
            # if csv_pred exists, read it
            start_time = time.time()
            if (csv_pred_prefix is not None) and os.path.exists(csv_pred) and (not force_rerun):
                print(f'{csv_pred} found, reading it')
                y_df = pd.read_csv(csv_pred,index_col=0)
                # drop nan
                y_df = y_df.dropna()
                y_test = y_df['y_test']
                y_pred = y_df['y_pred']
                

            else: 

                if summary_only:
                    print(f'summary_only is True, and {csv_pred} not found. Skipping {group_label}={group_value}')
                else:
                    if model is not None:
                        try:
                            model.fit(X_train, y_train, test=(X_test, y_test))
                        except Exception as e:
                            model.fit(X_train, y_train)                    
                        y_pred = model.predict(X_test)               

                        if (csv_pred_prefix is not None) and (not os.path.exists(csv_pred)):
                            y_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}, index=y_test.index)
                            y_df.to_csv(csv_pred)      
                            
                    else: # @llm
                        X_train,y_train,X_valid,y_valid,X_test,y_test = X_train[0:int(0.9*len(X_train))],y_train[0:int(0.9*len(y_train))],X_train[int(0.9*len(X_train)):len(X_train)],y_train[int(0.9*len(y_train)):len(y_train)],X_test[0:len(X_test)],y_test[0:len(y_test)]
                        y_pred, learning_curve = train_and_predict_main(X_train, y_train, X_valid, y_valid, X_test, y_test)
                        learning_curve.to_csv(csv_pred.replace('.csv', '_learning_curve.csv'))
                        y_df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred}, index=y_test.index)
                        y_df.to_csv(csv_pred)
                              
            if summary_only and (not os.path.exists(csv_pred)):
                mad, std, maes, rmse, r2, pearson_r, pearson_p_value, spearman_r, spearman_p_value, kendall_r, kendall_p_value = [None] * (len(columns)-2)
            else:
                mad, std, maes, rmse, r2, pearson_r, pearson_p_value, spearman_r, spearman_p_value, kendall_r, kendall_p_value = get_scores_from_pred(y_test,y_pred)
                time_elapsed = time.time() - start_time
                print('')
                print(f'{group_label}={str(group_value)}: {n_entries}, {mad:.3f}, {std:.3f}')
                print(f'Test scores: MAE={maes:.3f}, RMSE={rmse:.3f}, R2={r2:.3f}')       
                print(f'Pearson: r={pearson_r:.3f}, p={pearson_p_value:.3f}')
                print(f'Spearman: r={spearman_r:.3f}, p={spearman_p_value:.3f}')
                print(f'Kendall: r={kendall_r:.3f}, p={kendall_p_value:.3f}')
                print(f'Time elapsed: {time_elapsed:.3f} s')
                print('',flush=True)
            
            if save_data_files: # used for debugging
                pd.concat([X_train,y_train],axis=1).to_csv(f'{csv_pred_prefix}_{group_value_}_train_data.csv')
                pd.concat([X_test,y_test],axis=1).to_csv(f'{csv_pred_prefix}_{group_value_}_test_data.csv')
                print(f'Saved {csv_pred_prefix}_{group_value_}_train_data.csv and {csv_pred_prefix}_{group_value_}_test_data.csv')

            
        scores.append([n_entries, ratio_entries, mad, std, maes, rmse, r2,
                       pearson_r, pearson_p_value, spearman_r, spearman_p_value, kendall_r, kendall_p_value])
        # scores_train.append([n_entries, ratio_entries, mad_train, std_train, maes_train, rmse_train, r2_train,
        #                   pearson_r_train, pearson_p_value_train, spearman_r_train, spearman_p_value_train, kendall_r_train, kendall_p_value_train])
        if baseline_csv is not None:
            scores_test.append([n_entries, ratio_entries, mad, std, maes_test, rmse_test, r2_test, pearson_r_test, pearson_p_value_test, spearman_r_test, spearman_p_value_test, kendall_r_test, kendall_p_value_test])
    # Convert to df    
    scores = pd.DataFrame(
        scores,
        index=group_value_list,
        columns=columns
        )
    scores.index.name=group_label

    # scores_train = pd.DataFrame(
    #     scores_train,
    #     index=group_value_list,
    #     columns=columns
    #     )
    # scores_train.index.name=group_label

    scores_test = pd.DataFrame(
        scores_test,
        index=group_value_list,
        columns=columns
        )
    scores_test.index.name=group_label

    if csv_file is not None:
        scores.to_csv(csv_file)
        # scores_train.to_csv(csv_file.replace('.csv','_train.csv'))
        if baseline_csv is not None:
            scores_test.to_csv(csv_file.replace('.csv','_test.csv'))

    return scores


def extract_embeddings(model, g):    
    import torch
    """
    Extract embeddings from the ALIGNN model.

    g: The input graph or tuple of graphs (main graph and line graph).
    """
    from sklearnutils import graph_to_line_graph

    if len(model.alignn_layers) == 0:
        raise ValueError("ALIGNN layers are not present.")
    
    # Check and set the device
    device = model.device   


    if not isinstance(g, tuple):
        lg = graph_to_line_graph(g)
    else:
        g, lg = g
     
    lg = lg.local_var()
    lg = lg.to(device)

    # angle features (fixed)
    z = model.angle_embedding(lg.edata.pop("h"))

    g = g.local_var()
    g = g.to(device)

    # initial node features: atom feature network
    x = g.ndata.pop("atom_features")
    x = model.atom_embedding(x)

    # initial bond features
    bondlength = torch.norm(g.edata.pop("r"), dim=1)
    y = model.edge_embedding(bondlength)

    # ALIGNN updates: update node, edge, triplet features
    for alignn_layer in model.alignn_layers:
        x, y, z = alignn_layer(g, lg, x, y, z)

    # gated GCN updates: update node, edge features
    for gcn_layer in model.gcn_layers:
        x, y = gcn_layer(g, x, y)

    # norm-activation-pool-classify
    h = model.readout(g, x)

    h = h.detach().cpu().numpy().squeeze()
    return h



def separate_contrib(X_train,y_train, X_test,y_test, 
                     X_test2=None, y_test2=None,
                     struct_compo='none'
                     ):
    import shap

    def print_performance(model,X_test,y_test):
        y_pred = model.predict(X_test)
        mad, _, maes, _, r2, _, _, _, _, _, _ =get_scores_from_pred(y_test,y_pred)
        print(f'MAE: {maes:.3f}, MAE/MAD: {maes/mad:.3f}, R2: {r2:.3f}')

    model1 = get_model('xgb')

    model1.fit(X_train,y_train)
    print('Performance on test set 1')
    print_performance(model1,X_test,y_test)  

    model2 = get_model('xgb')

    X_train2, y_train2 = pd.concat([X_train,X_test]), pd.concat([y_train,y_test])
    dy_train2 = y_train2 - model1.predict(X_train2)
    model2.fit(X_train2,dy_train2)


    feature_labels = get_matminer_feature_labels(struct_compo=struct_compo)
    compo_features = [i for i,feature in enumerate(X_train.columns) if feature in feature_labels['compo']]
    struc_features = [i for i,feature in enumerate(X_train.columns) if feature in feature_labels['struc']]

    explainer2 = shap.TreeExplainer(model2)

    if X_test2 is None: 
        print('Calculating SHAP on the test set 1:')        
        shap_values2 = explainer2(X_test)

    else:
        print('Calculating SHAP on the test set 2:')        
        shap_values2 = explainer2(X_test2)

        if y_test2 is not None:
            print('Before training on test set 1:')
            print_performance(model2,X_test2,y_test2)

            print('After training on test set 1:')
            y_pred = model2.predict(X_test2) + model1.predict(X_test2)
            mad, _, maes, _, r2, _, _, _, _, _, _ =get_scores_from_pred(y_test2,y_pred)
            print(f'MAE: {maes:.3f}, MAE/MAD: {maes/mad:.3f}, R2: {r2:.3f}')

    shap_values2_compo = shap_values2.values[:,compo_features]
    shap_values2_struc = shap_values2.values[:,struc_features]

    return shap_values2_compo, shap_values2_struc

#%%
def get_matminer_feature_labels(struct_compo = 'none'):
    from matminer.featurizers.base import MultipleFeaturizer
    from matminer.featurizers.composition import (ElementProperty, 
                                                    Stoichiometry, 
                                                    ValenceOrbital, 
                                                    IonProperty)

    from matminer.featurizers.structure import (SiteStatsFingerprint, 
                                                StructuralHeterogeneity,
                                                ChemicalOrdering, 
                                                StructureComposition, 
                                                MaximumPackingEfficiency)   
    # 128 structural feature
    struc_feat = [
        SiteStatsFingerprint.from_preset("CoordinationNumber_ward-prb-2017"), 
        StructuralHeterogeneity(),
        MaximumPackingEfficiency(),
        ChemicalOrdering()
        ]       
    # 145 compositional features
    compo_feat = [
        StructureComposition(Stoichiometry()),
        StructureComposition(ElementProperty.from_preset("magpie")),
        StructureComposition(ValenceOrbital(props=['frac'])),
        StructureComposition(IonProperty(fast=True))
        ]

    if struct_compo == 'compo':
        compo_feat.append(
            SiteStatsFingerprint.from_preset("LocalPropertyDifference_ward-prb-2017"),
        )
    elif struct_compo == 'struc':
        struc_feat.append(
            StructureComposition(Stoichiometry()),
        )
    elif struct_compo == 'none':
        pass

    feature_labels = {
        'compo': MultipleFeaturizer(compo_feat).feature_labels() ,
        'struc': MultipleFeaturizer(struc_feat).feature_labels() ,
    }
    return feature_labels
#%%
