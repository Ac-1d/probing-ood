import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Set2Set
from torch_geometric.data import Data, DataLoader
import torch_scatter

class FeedForward(nn.Module):
    """
    This class defines a simple feed-forward neural network.

    Attributes:
        layers (ModuleList): A list of linear layers.
        norm_layers (ModuleList): A list of batch normalization layers applied after each linear layer.
    """
    def __init__(self, input_dim, hidden_dims):
        super(FeedForward, self).__init__()
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim

    def forward(self, x):
        """
        Forward pass for the FeedForward model.

        Parameters:
            x (Tensor): Input tensor.
        """
        for layer, norm_layer in zip(self.layers, self.norm_layers):
            x = F.gelu(norm_layer(layer(x)))
        return x

class Set2SetModel(nn.Module):
    """
    This class implements a model that uses the Set2Set algorithm. This model is primarily used to handle sets of varying sizes.

    Attributes:
        set2set (Set2Set): The Set2Set layer that converts sets of varying sizes to a fixed size tensor.
    """
    def __init__(self, input_dim, processing_steps=3, num_layers=1):
        super(Set2SetModel, self).__init__()
        self.set2set = Set2Set(input_dim, processing_steps=processing_steps, num_layers=num_layers)

    def forward(self, x, batch):
        """
        Forward pass for the Set2Set model.

        Parameters:
            x (Tensor): Input tensor.
            batch (LongTensor): Batch vector which assigns each node to a specific example.
        """
        return self.set2set(x, batch)


class PredictorBandgap(nn.Module):
    """
    This class defines a model to predict the bandgap of a material.

    Attributes:
        layers (ModuleList): A list of linear layers.
        norm_layers (ModuleList): A list of batch normalization layers applied after each linear layer.
        out_layer (Linear): Output layer for final prediction.
    """
    def __init__(self, input_dim, hidden_dims):
        super(PredictorBandgap, self).__init__()
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim

        self.out_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Forward pass for the Bandgap Predictor model.

        Parameters:
            x (Tensor): Input tensor.
        """
        for layer, norm_layer in zip(self.layers, self.norm_layers):
            x = F.gelu(norm_layer(layer(x)))
        return F.softplus(self.out_layer(x))


PredictorPositive = PredictorBandgap

class Predictor(nn.Module):
    """
    This class defines a basic model to predict a property of a material.

    Attributes:
        layers (ModuleList): A list of linear layers.
        norm_layers (ModuleList): A list of batch normalization layers applied after each linear layer.
        out_layer (Linear): Output layer for final prediction.
    """
    def __init__(self, input_dim, hidden_dims):
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()

        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.norm_layers.append(nn.BatchNorm1d(hidden_dim))
            input_dim = hidden_dim

        self.out_layer = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        Forward pass for the Predictor model.

        Parameters:
            x (Tensor): Input tensor.
        """
        for layer, norm_layer in zip(self.layers, self.norm_layers):
            x = F.gelu(norm_layer(layer(x)))
        return self.out_layer(x)
        # return x
    




class EformBandgapModel_CompressionSet2Set(nn.Module):
    """
    This class defines a combined model to predict the formation energy and bandgap of a material.

    Attributes:
        compressor (FeedForward): The feed-forward neural network that compresses the input.
        set2set (Set2SetModel): The Set2Set model that converts sets of varying sizes to a fixed size tensor.
        predictor1 (Predictor): The model used to predict the formation energy.
        predictor2 (PredictorBandgap): The model used to predict the bandgap.
    """
    def __init__(self, input_dim, compressor_hidden_dim, predictor_hidden_dim1, predictor_hidden_dim2, processing_steps=3, num_layers=1):
        super(EformBandgapModel_CompressionSet2Set, self).__init__()
        set2set_input_dim = compressor_hidden_dim[-1]
        self.compressor = FeedForward(input_dim, compressor_hidden_dim)
        self.set2set = Set2SetModel(set2set_input_dim, processing_steps=processing_steps, num_layers=num_layers)
        set2set_output_dim = self.set2set.set2set.out_channels
        self.predictor1 = Predictor(set2set_output_dim, predictor_hidden_dim1)
        self.predictor2 = PredictorBandgap(set2set_output_dim, predictor_hidden_dim2)

    def forward(self, data):
        """
        Forward pass for the Combined model.

        Parameters:
            data (Data): Input data.
        """
        x = self.compressor(data.x)
        x = self.set2set(x, data.batch)
        return self.predictor1(x), self.predictor2(x)

    def count_parameters(self):
        """
        Count the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




class EformModel_CompressionSet2Set(nn.Module):
    """
    This class defines a combined model to predict the formation energy of a material.

    Attributes:
        compressor (FeedForward): The feed-forward neural network that compresses the input.
        set2set (Set2SetModel): The Set2Set model that converts sets of varying sizes to a fixed size tensor.
        predictor (Predictor): The model used to predict the formation energy.
    """
    def __init__(self, input_dim, compressor_hidden_dim, predictor_hidden_dim, processing_steps=3, num_layers=1):
        super(EformModel_CompressionSet2Set, self).__init__()
        set2set_input_dim = compressor_hidden_dim[-1]
        self.compressor = FeedForward(input_dim, compressor_hidden_dim)
        self.set2set = Set2SetModel(set2set_input_dim, processing_steps=processing_steps, num_layers=num_layers)
        set2set_output_dim = self.set2set.set2set.out_channels
        self.predictor = Predictor(set2set_output_dim, predictor_hidden_dim)

    def forward(self, data):
        """
        Forward pass for the Combined model.

        Parameters:
            data (Data): Input data.
        """
        x = self.compressor(data.x)
        x = self.set2set(x, data.batch)
        return self.predictor(x)

    def count_parameters(self):
        """
        Count the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



class BandgapModel_CompressionSet2Set(nn.Module):
    """
    This class defines a combined model to predict the bandgap of a material.

    Attributes:
        compressor (FeedForward): The feed-forward neural network that compresses the input.
        set2set (Set2SetModel): The Set2Set model that converts sets of varying sizes to a fixed size tensor.
        predictor (PredictorBandgap): The model used to predict the bandgap.
    """
    def __init__(self, input_dim, compressor_hidden_dim, predictor_hidden_dim, processing_steps=3, num_layers=1):
        super(BandgapModel_CompressionSet2Set, self).__init__()
        set2set_input_dim = compressor_hidden_dim[-1]
        self.compressor = FeedForward(input_dim, compressor_hidden_dim)
        self.set2set = Set2SetModel(set2set_input_dim, processing_steps=processing_steps, num_layers=num_layers)
        set2set_output_dim = self.set2set.set2set.out_channels
        self.predictor = PredictorBandgap(set2set_output_dim, predictor_hidden_dim)

    def forward(self, data):
        """
        Forward pass for the Combined model.

        Parameters:
            data (Data): Input data.
        """
        x = self.compressor(data.x)
        x = self.set2set(x, data.batch)
        return self.predictor(x)

    def count_parameters(self):
        """
        Count the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)









class Model_CompressionSet2Set(nn.Module):
    """
    This class defines a combined model to predict the formation energy of a material.

    Attributes:
        compressor (FeedForward): The feed-forward neural network that compresses the input.
        set2set (Set2SetModel): The Set2Set model that converts sets of varying sizes to a fixed size tensor.
        predictor (Predictor): The model used to predict the formation energy.
    """
    def __init__(self, input_dim, compressor_hidden_dim, predictor_hidden_dim, processing_steps=3, num_layers=1,positive_output=False):
        super(Model_CompressionSet2Set, self).__init__()
        set2set_input_dim = compressor_hidden_dim[-1]
        self.compressor = FeedForward(input_dim, compressor_hidden_dim)
        self.set2set = Set2SetModel(set2set_input_dim, processing_steps=processing_steps, num_layers=num_layers)
        set2set_output_dim = self.set2set.set2set.out_channels
        if positive_output:
            self.predictor = PredictorPositive(set2set_output_dim, predictor_hidden_dim)
        else:
            self.predictor = Predictor(set2set_output_dim, predictor_hidden_dim)

    def forward(self, data):
        """
        Forward pass for the Combined model.

        Parameters:
            data (Data): Input data.
        """
        x = self.compressor(data.x)
        x = self.set2set(x, data.batch)
        return self.predictor(x)

    def count_parameters(self):
        """
        Count the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
class Model_SimpleSum(nn.Module):
    """
    A simple model to predict the formation energy of a material from a set of vectors.
    Attributes:
        predictor (Predictor): A neural network to predict the energy contribution of each vector in the set.
    """
    def __init__(self, input_dim, hidden_dims):
        super(Model_SimpleSum, self).__init__()
        # Initialize the predictor which will be applied to each vector in the input set
        self.predictor = Predictor(input_dim, hidden_dims)

    def forward(self, x, batch=None):
        """
        Forward pass for the Model_SimpleSum model.
        Parameters:
            x (Tensor): Input tensor, shape [N, input_dim], where N is the total number of vectors in the batch.
            batch (LongTensor, optional): Batch vector which assigns each vector to a specific sample in the batch. 
                                          If None, it's assumed the input does not need batching.
        Returns:
            Tensor: The predicted formation energy for each sample in the batch.
        """

        # Apply the predictor to each vector in the input set
        individual_contributions = self.predictor(x.x)
        batch=x.batch

        # If batch is None, simply sum up all the contributions
        if batch is None:
            total_energy = individual_contributions.sum()
        else:
            # Use torch_scatter to sum the contributions for each sample in the batch
            total_energy = torch_scatter.scatter_add(individual_contributions.squeeze(), batch, dim=0)

        return total_energy
    
    def count_parameters(self):
        """
        Count the total number of trainable parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)    

