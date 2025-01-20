#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

#%%
import numpy as np
import pandas as pd
import pickle
import torch
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.data import Data, DataLoader
import torch.optim as optim
import os
from tqdm import tqdm

from gmp_snn import Model_CompressionSet2Set, Model_SimpleSum #EformBandgapModel_CompressionSet2Set, BandgapModel_CompressionSet2Set, EformModel_CompressionSet2Set

torch.set_default_tensor_type(torch.DoubleTensor)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#%%
def log(file_path, message):
    try:
        with open(file_path, 'a') as file:
            file.write(message + '\n')
    except IOError as e:
        print(f"Error: {e}")




def get_gmp_features(
    structures,
    nsigmas,
    MCSH_order,
    width,
    pkl_file,
    converter = None,
    structure_names = None,
    psp_path = "./QE-kjpaw.gpsp",
):
    from GMPFeaturizer import GMPFeaturizer, PymatgenStructureConverter
    if converter is None:
        converter = PymatgenStructureConverter()


    num_features = nsigmas * (MCSH_order+1) + 1
    print(f'num_features: {num_features}')

    sigmas = np.round(np.linspace(0.0,width,nsigmas+1,endpoint=True),4)[1:].tolist()
    orders = [-1] + list(range(MCSH_order +1))
    GMPs = {
        "GMPs": {   
            "orders": orders, 
            "sigmas": sigmas  
        },
        # path to the pseudo potential file
        "psp_path": psp_path,
        # basically the accuracy of the resulting features
        "overlap_threshold": 1e-12, 
        "scaling_mode": "both",
    }
    featurizer = GMPFeaturizer(GMPs=GMPs, converter=converter, calc_derivatives=False, verbose=True)
    features_raw = featurizer.prepare_features(structures, cores=0)
    features_list = [entry["features"] for entry in features_raw]
    df_features = pd.Series(features_list,index=structure_names)
    df_features.to_pickle(pkl_file)
    return features_list, df_features

#%%
def prepare_dataset(X, y=None):
    dataset = []
    if y is None: # test set
        for features in X:
            features = torch.tensor(features)
            data = Data(x=features)
            dataset.append(data)
        return dataset
    else: # training set
        for features, target in zip(X, y):
            features = torch.tensor(features)
            target = torch.tensor([target])
            data = Data(x=features, y=(target))
            dataset.append(data)
        return dataset


def prepare_dataloader(X, y, batch_size, val_ratio,val=None,test=None):

    dataset = prepare_dataset(X, y)

    if y is not None: # training set
        if val is None: # X,y is training-val set
            print("No validation set is provided. Splitting training set into training and validation sets.")
            print(f"Validation ratio: {val_ratio}")
            train_size = int((1-val_ratio) * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
        else: # X,y is training set, val is validation set
            train_loader = DataLoader(dataset, batch_size=batch_size)
            val_loader = DataLoader(prepare_dataset(*val), batch_size=batch_size)

        if test is not None:
            test_loader = DataLoader(prepare_dataset(*test), batch_size=batch_size)
        else:
            test_loader = None
        return train_loader, val_loader, test_loader
        
    else: # test set
        test_loader = DataLoader(dataset, batch_size=batch_size)
        return test_loader


def _fit(model,
        train_loader,
        val_loader,
        test_loader=None,
        scheduler=None,
              ):

    device = model.device
    n_epochs = model.n_epochs
    path = model.path

    ini_lr = 2e-3
    optimizer = optim.Adam(model.parameters(), lr=ini_lr)

    if scheduler is None:
        patience = max(10,int(n_epochs/15)) # no smaller than 10
        patience = min(25,patience)  # no larger than 25
        factor = 0.5
        threshold = 1e-4 #
        min_lr = 1e-4
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience, threshold=threshold, factor=factor,min_lr=min_lr)
        message = f"Scheduler: ReduceLROnPlateau, patience={patience}, threshold={threshold}, factor={factor}, min_lr={min_lr}" 
        log(path+"/log.dat", message)   

    model = model.to(device)
    criterion = nn.L1Loss()
    best_val_loss = float('inf')
    # Training loop
    for epoch in range(n_epochs):

        # Training
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch)  # Forward pass
            loss = criterion(pred.view(-1), batch.y)
            loss.backward()  
            optimizer.step() 
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(train_loader.dataset)    


        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader):
                batch = batch.to(device)
                pred = model(batch)
                loss = criterion(pred.view(-1), batch.y)
                val_loss += loss.item() * batch.num_graphs
            val_loss /= len(val_loader.dataset)
            
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        
        if test_loader is None:
            loss_info = f"Epoch: {epoch+1}, Target: {loss:.4f}, Train Loss: {total_loss:.5f}, Val Loss: {val_loss:.5f}, LR: {current_lr}"
        else:
            # Test set evaluation
            model.eval()
            test_loss = 0
            with torch.no_grad():
                for batch in tqdm(test_loader):
                    batch = batch.to(device)
                    pred = model(batch)
                    loss = criterion(pred.view(-1), batch.y)
                    test_loss += loss.item() * batch.num_graphs
                test_loss /= len(test_loader.dataset)

            loss_info = f"Epoch: {epoch+1}, Target: {loss:.4f}, Train Loss: {total_loss:.5f}, Val Loss: {val_loss:.5f}, Test Loss: {test_loss:.5f}, LR: {current_lr}"


        print(loss_info)
        
        # Learning rate scheduler step
        scheduler.step(val_loss)
        
        # Save model if validation loss has decreased
        if val_loss < best_val_loss:
            print("model saved")
            loss_info += "\t\t+ "
            best_val_loss = val_loss
            try:
                torch.save(model.state_dict(), os.path.join(path,'combined.pth'))
                torch.save(model.set2set.state_dict(), os.path.join(path,'set2set.pth'))
                torch.save(model.predictor.state_dict(), os.path.join(path,'predictor.pth'))
                torch.save(model.compressor.state_dict(), os.path.join(path,'compressor.pth'))
            except:
                pass
        log(path+"/log.dat", loss_info)    


def _predict(model, test_loader,index):
    device = model.device
    model.eval()
    y_pred = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            pred = model(batch)
            y_pred.append(pred.view(-1).cpu().numpy())
    y_pred = np.concatenate(y_pred)
    # convert to pandas Series
    y_pred = pd.Series(y_pred, index=index)
    return y_pred


# maybe use a more general model class instead of EformModel_CompressionSet2Set
class skModel_CompressionSet2Set(Model_CompressionSet2Set):
    def __init__(self, 
                 input_dim, compressor_hidden_dim, predictor_hidden_dims, processing_steps, num_layers,positive_output,
                 n_epochs,
                 path,
                 val_ratio=0.1, batch_size=512,reset_parameters=True, ):
        
        super().__init__(input_dim, compressor_hidden_dim, predictor_hidden_dims, processing_steps, num_layers,positive_output)
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.path = path
        self.device = device
        self.reset_parameters = reset_parameters
        
        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.state_dict(), path+'/model_initial.pth')

        # save model info
        model_info = "*"*40 + "\n"
        model_info += "input_dim: {}\n".format(input_dim)
        model_info += "compressor_hidden_dim: {}\n".format(','.join(map(str, compressor_hidden_dim)))
        model_info += "predictor_hidden_dim: {}\n".format(','.join(map(str, predictor_hidden_dims)))
        model_info += "set2set_processing_steps: {}\n".format(processing_steps)
        model_info += "set2set_num_layers: {}\n".format(num_layers)
        model_info += "num_parameters: {}\n".format(self.count_parameters())
        model_info += "*"*40
        log(path+"/info.dat", model_info)


    def fit(self, X, y,val=None,test=None):
        if self.reset_parameters:
            path = self.path
            self.load_state_dict(torch.load(path+'/model_initial.pth'))

        train_loader, val_loader, test_loader = prepare_dataloader(X, y, self.batch_size, self.val_ratio,val,test)
            
        _fit(self, train_loader, val_loader, test_loader)

    def predict(self, X):
        index = X.index
        test_loader = prepare_dataloader(X, None, self.batch_size, self.val_ratio)
        return _predict(self, test_loader,index)



class skModel_SimpleSum(Model_SimpleSum):
    def __init__(self, 
                 input_dim, hidden_dim, 
                 n_epochs,
                 path,
                 val_ratio=0.1, batch_size=512,reset_parameters=True, ):

        super().__init__(input_dim, hidden_dim)
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.path = path
        self.device = device
        self.reset_parameters = reset_parameters

        if not os.path.exists(path):
            os.makedirs(path)

        torch.save(self.state_dict(), path+'/model_initial.pth')

        # save model info
        model_info = "*"*40 + "\n"
        model_info += "input_dim: {}\n".format(input_dim)
        model_info += "hidden_dim: {}\n".format(','.join(map(str, hidden_dim)))
        model_info += "num_parameters: {}\n".format(self.count_parameters())
        model_info += "*"*40
        log(path+"/info.dat", model_info)


    def fit(self, X, y,val=None,test=None):
        if self.reset_parameters:
            path = self.path
            self.load_state_dict(torch.load(path+'/model_initial.pth'))

        train_loader, val_loader, test_loader = prepare_dataloader(X, y, self.batch_size, self.val_ratio,val,test)
            
        _fit(self, train_loader, val_loader, test_loader)

    def predict(self, X):
        index = X.index
        test_loader = prepare_dataloader(X, None, self.batch_size, self.val_ratio)
        return _predict(self, test_loader,index)
    

