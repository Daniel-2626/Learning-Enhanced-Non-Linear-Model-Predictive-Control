import os
import torch
import torch.nn as nn
import torch.nn.functional as F # a bunch of functions, e.g. convolution
import pandas as pd
import numpy as np
from collections import defaultdict # https://docs.python.org/3/library/collections.html#collections.defaultdict
from torch.func import functional_call
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.signal import savgol_filter
from scipy import signal
import random
import time

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Load dataset of residuals
csv_path = os.path.join(os.path.dirname(__file__), 'cascaded_nominal_matched.csv')
df = pd.read_csv(csv_path)

inputs = df[['h1','h2','u']].to_numpy()
X_train = inputs
# Precomputed residual already
Y_target = df[['residual_1', 'residual_2']].to_numpy()

### MLP ARCHITECTURE ###
# input_dim, output_dim, hidden_dim, num_layers will be changed lower in the code
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=3):
        super(MLP, self).__init__()
        self.tau = 2
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()] # nn.Tanh hyperbolic tangent
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]) # nn.Linear applies an affine transform
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        net =  self.net(x)
        # Bound NN output by tanh and confidence parameter tau
        return self.tau* torch.tanh(net)

# Training 
def main():
    device = torch.device("cpu")
    
    # Hyperparameters (Must have the same input_dim, output_dim, hidden_dim and num_layers as being used in the adaptive)
    learning_rate = 1e-3
    input_dim = 3
    output_dim = 2
    hidden_dim = 8
    num_layers = 2

    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    residual_mlp = MLP(input_dim = input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers) # the network
    start = time.time()

    residual_mlp = residual_mlp
    residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=learning_rate, weight_decay=0.1) # lr = learning rate, the optimizer
    
    residual_criterion = nn.MSELoss()
    X_batch = torch.tensor(X_train, dtype=torch.float32)
    y_target = torch.tensor(Y_target, dtype=torch.float32)
    
    for p in residual_mlp.parameters(): p.requires_grad = True

    # Epoch loop
    for _ in range(200):
        residual_optimizer.zero_grad() 
        prediction = residual_mlp(X_batch) 

        # Minimize (plant - (nominal + residual))^2
        loss = residual_criterion(prediction, y_target)
        loss.backward() 
        residual_optimizer.step() 
    end = time.time()

    # Saving trained NN parameters
    torch.save(residual_mlp.state_dict(), "REMOVE_TESTcascaded_tanks_deriv_pretrain.pth")

    print("elapsed", 1000*(end-start))

if __name__ == "__main__":
    main()
    

