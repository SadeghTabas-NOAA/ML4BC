import torch
import torch.nn as nn
import netCDF4 as nc
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from datetime import datetime, timedelta, date
import xarray as xr
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adjacency_matrix):
        """
        x: Input features (batch_size, num_nodes, input_features)
        adjacency_matrix: Graph adjacency matrix (batch_size, num_nodes, num_nodes)
        """
        # Graph convolution for each batch
        x = torch.matmul(adjacency_matrix, x.transpose(1, 2)).transpose(1, 2)
        x = self.linear(x)  # Linear layer
        x = F.relu(x)  # ReLU activation
        return x

class GraphTemporalModel(nn.Module):
    def __init__(self, num_variables, num_nodes, num_time_steps, hidden_size=32, num_layers=2):
        super(GraphTemporalModel, self).__init__()

        self.num_variables = num_variables
        self.num_nodes = num_nodes
        self.num_time_steps = num_time_steps

        # Graph convolutional layers
        self.graph_conv1 = GraphConvolution(self.num_variables, 16)
        self.graph_conv2 = GraphConvolution(16, 8)

        # LSTM layer to capture temporal dependencies
        self.lstm = nn.LSTM(input_size=8, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, adjacency_matrix):
        """
        x: Input features (batch_size, num_variables, num_time_steps, num_nodes)
        adjacency_matrix: Graph adjacency matrix (batch_size, num_nodes, num_nodes)
        """

        # Reshape input for graph convolution
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(-1, self.num_variables, self.num_time_steps)

        # Apply graph convolution layers
        x = self.graph_conv1(x, adjacency_matrix)
        x = self.graph_conv2(x, adjacency_matrix)

        # Global average pooling
        x = x.mean([2])

        # Reshape for LSTM
        x = x.view(-1, self.num_time_steps, 8)

        # Apply LSTM
        x, _ = self.lstm(x)

        # Fully connected layer
        x = self.fc(x[:, -1, :])

        # Reshape to match output shape
        x = x.view(-1, self.num_nodes, 1)

        return x


# Example usage
num_variables = 10  # Change this based on your actual number of variables
num_nodes = 181 * 360  # Assuming spatial grid of 181x360
num_time_steps = 129
# Create the model
if torch.cuda.device_count() > 1:
    print("Using", torch.cuda.device_count(), "GPUs!")
    dlmodel = nn.DataParallel(GraphTemporalModel(num_variables, num_nodes, num_time_steps)).to(device)
else:
    dlmodel = GraphTemporalModel(num_variables, num_nodes, num_time_steps).to(device)
    