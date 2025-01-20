#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kangming
"""
#%%
from extra_funcs import leave_one_group_out, load_data,get_args,set_model_output_dir,get_scores_from_pred,get_split
from pymatgen.core.periodic_table import Element
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from myfunc import get_col2drop

dataset = 'mp21' # 'mp21','jarvis22','oqmd', # greater_than_nelements
modelname = 'xgb'
target = 'e_form'

pretrained = False #True False
pretrained_modelname = modelname

if pretrained:
    prefix = 'all' 

else:
    prefix = 'uncorr' #'uncorr_important','all','uncorr'

    cumulative_importance_cutoff = 0.95



#%%
if modelname == 'rf':
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
    n_estimators=100, max_features=1/3, n_jobs=-1, random_state=0
    )
elif modelname == 'lasso':
    from sklearn.linear_model import Lasso
    model = Lasso(alpha=0.01)
elif modelname == 'linear':
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
elif modelname == 'dt': # decision tree
    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()

modelnames = {
    #'llm':'LLM-Prop',
    'alignn25':'ALIGNN',
    'gmp':'GMP',
    'xgb':'XGB',
    'rf':'RF',}

#%%
df, X_matminer, y = load_data('xgb',dataset,target)

#%% @@
'''
Loading embeddings from models pretrained on leave-one-group-out datasets
'''

# group_label = 'elements'
# group_value = 'O'

# group_label = 'crystal_system'
# group_value = 'triclinic'

# group_label = 'greater_than_nelements'
# group_value = 4

group_label = 'period'
group_value = 5

fraction=''
# fraction = '1.0';rand_id = 0

#%%


# group_label = 'point_group'
# group_value = 'm-3m'

if pretrained:
    if pretrained_modelname == 'alignn25':
        if group_value == 'all': # pretrained on all
            logo_dir = f'embedding/trained_on_all/{pretrained_modelname}_{dataset}_{target}'
        else: # pretrained on leave-one-group-out
            if fraction == '':
                logo_dir = f'embedding/leave_one_group_out/{dataset}_{target}_{group_label}_{group_value}'
            else:
                # logo_dir = f'embedding/leave_one_group_out/fraction/{dataset}_{target}/{group_label}_{group_value}_{fraction}/{rand_id}'
                logo_dir = f'output_id/{pretrained_modelname}/{dataset}/{target}/{group_label}_{group_value}_{fraction}/{rand_id}' 
    elif pretrained_modelname == 'gmp':
        logo_dir = f'learning_curve/{dataset}_{group_label}_{group_value}_{target}_{pretrained_modelname}'
    print(f'Loading embeddings from {logo_dir}')

    embedding_filename = f'{logo_dir}/embeddings.json'
    if os.path.exists(embedding_filename):
        embeddings = pd.read_json(embedding_filename)
        print(f'Loaded {embedding_filename}')
    else:
        raise FileNotFoundError(f'Embedding file not found: {embedding_filename}')

    # expand the embeddings into a dataframe
    # embeddings = pd.DataFrame(embeddings.tolist(),index=embeddings.index)
    # embeddings.columns = [f'emb{i}' for i in range(embeddings.shape[1])]
    if pretrained_modelname == 'alignn25':
        embeddings = embeddings.T
    X_all = embeddings.loc[df.index]
else: 
    X_all = X_matminer

#%%
X = X_all
df.index = df.index.astype(str)
X.index = X.index.astype(str)
y.index = y.index.astype(str)

#%%
# standardize X 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X),columns=X.columns,index=X.index)

#%% Drop highly correlated features

# if 'uncorr' in prefix: # drop highly correlated features if prefix contains 'uncorr'

#     # remove columns with zero variance
#     X = X.loc[:,X.std()!=0]
#     # get the columns to keep and drop
#     col2keep, col2drop = get_col2drop(X,cutoff=0.7)
#     X = X[col2keep]
#     if prefix == 'uncorr_important':
#         model.fit(X,y)
#         # get the feature importance
#         feature_importance = pd.DataFrame(model.feature_importances_,index=X.columns,columns=['importance'])
#         feature_importance.sort_values(by='importance',ascending=False,inplace=True)
#         cumulative_importances = feature_importance['importance'].cumsum()
#         index = np.where(cumulative_importances >= cumulative_importance_cutoff)[0][0]
#         important_features = feature_importance.index[:index+1]
#         X = X[important_features]




#%% 
'''
UMAP
'''        

import matplotlib.pyplot as plt
from umap import UMAP

def get_umap_dat(dataset,supervised,n_neighbors,
                 X,y,
                 prefix='all',
                 target='',
                 densmap=False,
                 pretrained=None):

    if pretrained is None:
        emb = 'matminer'
    else:
        emb = pretrained

    if supervised:
        umap_csv = f'umap_csv/{dataset}_{emb}_{prefix}_{n_neighbors}nn_{target}_supervised.csv'
    else:
        if densmap:
            umap_csv = f'umap_csv/{dataset}_{emb}_{prefix}_{n_neighbors}nn_unsupervised_densmap.csv'
        else:
            umap_csv = f'umap_csv/{dataset}_{emb}_{prefix}_{n_neighbors}nn_unsupervised.csv'

    if os.path.exists(umap_csv):
        print(f'UMAP: {umap_csv} found, loading...')
        X_umap = pd.read_csv(umap_csv,index_col=0)
    else:
        print(f'UMAP: {umap_csv} not found, calculating...')
        umap = UMAP(n_components=2, n_neighbors=n_neighbors,densmap=densmap,dens_lambda=0.1)
        # use X to train umap
        if supervised:
            X_umap = umap.fit_transform(X,y=y)
        else:
            X_umap = umap.fit_transform(X)
        X_umap = pd.DataFrame(X_umap,columns=['0','1'],index=X.index)
        X_umap.to_csv(umap_csv)

    return X_umap


if 'alignn' in pretrained_modelname or 'gmp' in pretrained_modelname:
    if fraction == '':
        pretrained_filename = f'{pretrained_modelname}-{target}-{group_label}-{group_value}'
    else:
        pretrained_filename = f'{pretrained_modelname}-{target}-{group_label}-{group_value}{fraction}-{rand_id}'

else:
    pretrained_filename = None

for n_neighbors in [50]: # 50,100,
    for supervised in [False]:# [False,True]:
        X_umap = get_umap_dat(
            dataset,supervised,n_neighbors,X,y,prefix,target,
            densmap=False, # interestingly densmap is not better
            pretrained = pretrained_filename
            )


#%%
def get_index_given_element(df,element):
    if isinstance(element,list):
        has_element = df[df['elements'].apply(lambda x: any([e in x for e in element]))].index.tolist()
        dont_have_element = df[df['elements'].apply(lambda x: not any([e in x for e in element]))].index.tolist()
    else:
        has_element = df[df['elements'].apply(lambda x: element in x)].index.tolist()
        dont_have_element = df[df['elements'].apply(lambda x: element not in x)].index.tolist()
    return has_element,dont_have_element

def get_index_given_period(df,period):
    if isinstance(period,str):
        period = int(period)
    has_period = df[df['period'].apply(lambda x: period in x)].index.tolist()
    dont_have_period = df[df['period'].apply(lambda x: period not in x)].index.tolist()
    return has_period,dont_have_period


def get_index_given_group(df,group_label,group_value):
    is_group = df[df[group_label]==group_value].index.tolist()
    is_not_group = df[df[group_label]!=group_value].index.tolist()
    return is_group,is_not_group

def get_index_given_min_nelements(df,min_nelements):
    is_min_nelements = df[df['nelements']>min_nelements].index.tolist()
    is_not_min_nelements = df[df['nelements']<=min_nelements].index.tolist()
    return is_min_nelements,is_not_min_nelements


# group_label = 'greater_than_nelements'
# group_value = 4
# group_label = 'period'
# group_value = 5

# group_label = 'elements'
# group_value = 'H'

# if group_label == 'elements':
#     element = group_value
#     index_has_element,index_dont_have_element = get_index_given_element(df,group_value)
#     X_train, y_train = X.loc[index_dont_have_element], y.loc[index_dont_have_element]
#     X_test, y_test = X.loc[index_has_element], y.loc[index_has_element]
# elif group_label == 'period':
#     period = group_value
#     index_has_period,index_dont_have_period = get_index_given_period(df,group_value)
#     X_train, y_train = X.loc[index_dont_have_period], y.loc[index_dont_have_period]
#     X_test, y_test = X.loc[index_has_period], y.loc[index_has_period]
# elif group_label in ['point_group']:
#     index_is_group,index_is_not_group = get_index_given_group(df,group_label,group_value)
#     X_train, y_train = X.loc[index_is_not_group], y.loc[index_is_not_group]
#     X_test, y_test = X.loc[index_is_group], y.loc[index_is_group]
# elif group_label == 'greater_than_nelements':
#     is_min_nelements,is_not_min_nelements = get_index_given_min_nelements(df,group_value)
#     X_train, y_train = X.loc[is_not_min_nelements], y.loc[is_not_min_nelements]
#     X_test, y_test = X.loc[is_min_nelements], y.loc[is_min_nelements]
# elif group_label == 'le_nelements':
#     is_min_nelements,is_not_min_nelements = get_index_given_min_nelements(df,group_value)
#     X_train, y_train = X.loc[is_min_nelements], y.loc[is_min_nelements]
#     X_test, y_test = X.loc[is_not_min_nelements], y.loc[is_not_min_nelements]

index_train, index_test = get_split(df, group_label,group_value)
X_train, y_train = X.loc[index_train], y.loc[index_train]
X_test, y_test = X.loc[index_test], y.loc[index_test]

print(f'Leave out {group_label}={group_value}')
print(f'Train set size: {len(y_train)}')
print(f'Test set size: {len(y_test)}')


#%%
# model.fit(X_all.loc[X_train.index],y_train)
# y_pred = model.predict(X_all.loc[X_test.index])
# y_pred = pd.Series(y_pred,index=X_test.index)
# y_err = (y_pred - y_test).abs()
# y_err.mean()

#%% read from csv 
if fraction == '':
    df_y = pd.read_csv(
        f'leave_one_group_out/{target}/{dataset}_{pretrained_modelname}_leave_one_{group_label}_out_pred_{group_value}.csv',
        index_col=0)
else:
    df_y = pd.read_csv(f'{logo_dir}/y_pred_ood.csv',index_col=0)
df_y.index = df_y.index.astype(str)
# df_y = pd.read_csv(f'leave_one_group_out/{target}/{dataset}_xgb_leave_one_{group_label}_out_pred_{group_value}.csv',index_col=0)

df_y = df_y.loc[X_test.index]
y_test = df_y['y_test']
y_pred = df_y['y_pred']
y_err = (y_pred - y_test).abs()
y_err.mean()


#%%
from scipy.stats import gaussian_kde

# id_train = X_train.index.tolist()
# id_test = X_test.index.tolist()

# id_train = [str(i) for i in id_train]
# id_test = [str(i) for i in id_test]

# Convert the indices to string type

id_train = X_train.index
id_test = X_test.index
X_umap.index = X_umap.index.astype(str)

X_umap_train = X_umap.loc[id_train]
X_umap_test = X_umap.loc[id_test]

# Prepare data
x_train = X_umap_train.values.T
x_test = X_umap_test.values.T
# Perform KDE on X_train
kde = gaussian_kde(x_train, 
                   bw_method=0.01
                   )


#%%
from multiprocessing import Pool, cpu_count

# Worker function
def kde_worker(x_chunk):
    return kde(x_chunk)

# Split x_test into chunks
chunks = np.array_split(x_test, cpu_count(),axis=1)

# Create a multiprocessing Pool
with Pool() as pool:
    # Calculate kde for each chunk
    results = pool.map(kde_worker, chunks)

# Combine the results
density = np.concatenate(results)
# Calculate density of X_train at each point of X_test
# density = kde(x_test)
# convert density to pandas Series
density = pd.Series(density,index=X_test.index)


#%%

xmax = 0.04
dx = 0.001

mean_err = []
std_err = []
x_1 = np.arange(0,0.004,0.001)
x_2 = np.arange(0.004,0.02,0.003)
x_3 = np.arange(0.02,0.045,0.005)
x = np.concatenate([x_1,x_2,x_3])
x_index = []

for index, i in enumerate(x[:-1]):
    data = y_err[density.between(x[index],x[index+1])]
    x_index.append((x[index]+x[index+1])/2)
    mean_err.append(data.mean())
    std_err.append(data.std())

x_index = np.array(x_index)
df_err = pd.DataFrame({'mean_err':mean_err,'std_err':std_err},index=x[:-1])

fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(density,y_err,s=25,alpha=0.1,c='green')

ax.set_xlabel('Kernel-density estimate')
ax.set_ylabel(r'$|y_{\rm pred} - y_{\rm true}|$ (eV/atom)')
ax.set_xlim(0,xmax)
ax.set_ylim(0,3)
# set yticks
yticks = np.arange(0,4.,1)
ax.set_yticks(yticks)

# make a second y-axis on the right side
ax2 = ax.twinx()
ax2.set_ylabel('Mean absolute error (eV/atom)')
# ax2.errorbar(df_err.index+dx/2,df_err['mean_err'],yerr=df_err['std_err'],fmt='o',color='black')
ax2.plot(x_index,
         df_err['mean_err'],'o-', color='black',markerfacecolor='white')
# set yticks
yticks = np.arange(0.1,0.3,0.05)
ax2.set_yticks(yticks)
ax2.set_ylim(yticks[0],yticks[-1])




figname = f'figs/paper/{dataset}_{pretrained_modelname}_{prefix}_{n_neighbors}nn_{group_label}_{group_value}_density.png'
fig.savefig(
    figname,
    transparent=False,facecolor='white',edgecolor='white',
    dpi=300,bbox_inches='tight',
    )


#%% @@@
# Define density threshold
threshold = 0.001 

# Create a boolean mask based on the density threshold
mask = density > threshold

(mad, std, maes, rmse, r2, pearson_r, pearson_p_value, 
    spearman_r, spearman_p_value, kendall_r, kendall_p_value) = get_scores_from_pred(
        y_test,y_pred)
print(f'Test: entries={len(y_pred)}, mae={maes:.3f}, mae/mad={maes/mad:.3f}, r2={r2:.3f}, pearson_r={pearson_r:.3f}')
r2_all = r2

# get the scores for low density and high density test errors
(mad, std, maes, rmse, r2, pearson_r, pearson_p_value, 
    spearman_r, spearman_p_value, kendall_r, kendall_p_value) = get_scores_from_pred(
        y_test[mask],y_pred[mask])
print(f'HD test: entries={len(y_pred[mask])}, mae={maes:.3f}, mae/mad={maes/mad:.3f}, r2={r2:.3f}, pearson_r={pearson_r:.3f}')
maes_hd = maes
r2_hd = r2
(mad, std, maes, rmse, r2, pearson_r, pearson_p_value, 
    spearman_r, spearman_p_value, kendall_r, kendall_p_value) = get_scores_from_pred(y_test[~mask],y_pred[~mask])
print(f'LD test: entries={len(y_pred[~mask])}, mae={maes:.3f}, mae/mad={maes/mad:.3f}, r2={r2:.3f}, pearson_r={pearson_r:.3f}')
maes_ld = maes
r2_ld = r2

# y_test[mask].to_csv(f'{logo_dir}/y_test_hd.csv')
# y_test[~mask].to_csv(f'{logo_dir}/y_test_ld.csv')


#%%
alpha = 0.1
fig, ax = plt.subplots(figsize=(4.5,4.5))
ax.scatter(y_test[mask],y_pred[mask],alpha=alpha,color='blue',label='HD')
ax.scatter(y_test[~mask],y_pred[~mask],alpha=alpha,color='red',label='LD')
ax.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='black')
ax.set_xlabel('DFT')
ax.set_ylabel('ML')
ax.legend()

#%%

disp_y = 0; disp_x = 0 # for adjusting the position of the plot

alpha_factor = 0.2
alpha=0.1;s=6; fontsize=12

fig, ax = plt.subplots(figsize=(4.,4.))

dataset_name = dataset.split('2')[0].upper()

# ax.text(0.025,0.975,f'Leave {group_label.replace("s","")} {group_value} out ({dataset_name})',transform=ax.transAxes,ha='left',va='top',fontsize=fontsize)
# # ax.text(0.025,0.975,f'Leave $\geq$quinary out ({dataset_name})',transform=ax.transAxes,ha='left',va='top',fontsize=fontsize)
# ax.text(0.025,0.925,f'{modelnames[modelname]} $R^2$ (HD): {r2_hd:.3f}',transform=ax.transAxes,ha='left',va='top',fontsize=fontsize,color='blue')
# ax.text(0.025,0.875,f'{modelnames[modelname]} $R^2$ (LD): {r2_ld:.3f}',transform=ax.transAxes,ha='left',va='top',fontsize=fontsize,color='red')

ax.text(0.025,0.99,f'{modelnames[modelname]} $R^2$ on compounds with'+r' $\geq$5 elements',transform=ax.transAxes,ha='left',va='top',fontsize=fontsize-1.27)

# ax.text(0.025,0.99,f'{modelnames[modelname]} $R^2$ on period-5 compounds',transform=ax.transAxes,ha='left',va='top',fontsize=fontsize-1)
ax.text(0.025,0.935,f'All-domain: {r2_all:.3f}',transform=ax.transAxes,ha='left',va='top',fontsize=fontsize,color='k')
ax.text(0.025,0.885,f'In-domain: {r2_hd:.3f}',transform=ax.transAxes,ha='left',va='top',fontsize=fontsize,color='b')
ax.text(0.025,0.835,f'Out-of-domain: {r2_ld:.3f}',transform=ax.transAxes,ha='left',va='top',fontsize=fontsize,color='red')



# Plot X_train
ax.scatter(X_umap_train['0'], 
           X_umap_train['1'], 
           color='grey', s=s,
           label='Train',alpha=alpha)

# Plot X_test with the color map
X_umap_test_high = X_umap_test[mask]
X_umap_test_low = X_umap_test[~mask]
ax.scatter(X_umap_test_high['0'], 
           X_umap_test_high['1'], 
           c='blue', s=s*0.6,label='Test (HD)',alpha=alpha*alpha_factor)
ax.scatter(X_umap_test_low['0'], 
           X_umap_test_low['1'], 
           c='red', s=s*0.6,label='Test (LD)',alpha=alpha*alpha_factor)


rscale = 0.1
d = (X_umap['0'].max() - X_umap['0'].min())*rscale
xlims = np.array([X_umap['0'].min()-d/2,X_umap['0'].max()+d/2])
d = (X_umap['1'].max() - X_umap['1'].min())*rscale
ylims = np.array([X_umap['1'].min()-0.5*d,X_umap['1'].max()+1.25*d])
ax.set_xlim((xlims+disp_x))
# ax.set_xlim((xlims+disp_x)[::-1]) # reverse the x-axis
ax.set_ylim((ylims+disp_y))


# remove ticks
ax.set_xticks([])
ax.set_yticks([])

# # get ylims
# ylims = ax.get_ylim()
# ax.set_ylim(ylims[::-1])

# get xlims
xlims = ax.get_xlim()
ax.set_xlim(xlims[::-1])

# Create legend handles manually
train_patch = mlines.Line2D([], [], color='grey', marker='o', linestyle='None',
                        markersize=5, label='Train')
test_hd_patch = mlines.Line2D([], [], color='blue', marker='o', linestyle='None',
                        markersize=5, label='Test (HD)')
test_ld_patch = mlines.Line2D([], [], color='red', marker='o', linestyle='None',
                        markersize=5, label='Test (LD)')

# Add the legend to the plot
# ax.legend(handles=[train_patch, test_hd_patch, test_ld_patch],
#         #   loc='lower left',
#           loc='upper right',
#           fontsize=fontsize
#           )

if fraction == '':
    figname = f'figs/paper/{dataset}_{pretrained_modelname}_{prefix}_{n_neighbors}nn_{group_label}_{group_value}.png'
else:
    figname = f'figs/paper/{dataset}_{pretrained_modelname}_{prefix}_{n_neighbors}nn_{group_label}_{group_value}_{fraction}_{rand_id}.png'

fig.savefig(
    figname,
    # white background
    transparent=False,facecolor='white',edgecolor='white',
    # bbox_inches='tight',
    dpi=300,
    )


#%%