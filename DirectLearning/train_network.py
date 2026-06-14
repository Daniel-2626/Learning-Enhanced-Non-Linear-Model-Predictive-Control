import os
import torch
import torch.nn as nn
import torch.nn.functional as F # a bunch of functions, e.g. convolution
import numpy as np
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


# MLP model definition
class MLP(nn.Module):
    def __init__(self, input_dim=4, output_dim=2, hidden_dim=16, num_layers=3):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# MAML Training Loop
residual_mlp = MLP(input_dim = 2 + 1, output_dim=2, hidden_dim=16, num_layers=2) # the network
device = torch.device("cpu")
learning_rate = 1e-3
for param in residual_mlp.parameters():
    param.requires_grad = False
residual_mlp = residual_mlp
residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=learning_rate) # lr = learning rate, the optimizer
residual_criterion = nn.MSELoss()

def train_network(residual_data, input_data):
        
    X_batch = torch.tensor(input_data, dtype=torch.float32)
    y_target = torch.tensor(residual_data, dtype=torch.float32)

    for p in residual_mlp.parameters(): p.requires_grad = True
    for _ in range(200):
        residual_optimizer.zero_grad() # optimizer object
        prediction = residual_mlp(X_batch) # gives data to network to make a prediction
        loss = residual_criterion(prediction, y_target)
        l2_norm = sum(p.pow(2).sum() for p in residual_mlp.parameters())
        regularization = 0.1
        loss += regularization * l2_norm
        loss.backward() # calculates gradient
        residual_optimizer.step() # one optimization step to update parameters
    for p in residual_mlp.parameters(): p.requires_grad = False
    
    weights = np.array([])
    for p in residual_mlp.parameters():
        weights = np.append(weights, p.flatten())
    return weights

