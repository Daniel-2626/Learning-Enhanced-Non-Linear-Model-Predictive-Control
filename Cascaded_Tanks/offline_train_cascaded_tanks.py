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
csv_path = os.path.join(os.path.dirname(__file__), 'cascaded_residuals.csv')
df = pd.read_csv(csv_path)
print(df.head())

inputs = df[['h1','h2','u']].to_numpy()
X_train = inputs

#X_train = np.transpose(X_train)
print(X_train.shape)


Y_target = df[['residual_1', 'residual_2']].to_numpy()

#print(Y_target)
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=3):
        super(MLP, self).__init__()
        self.tau = 0.5
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()] # nn.ReLU rectified linear function (max(x,0))
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]) # nn.Linear applies an affine transform. hidden_dim features and hidden_dim out features
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        net =  self.net(x)
        return self.tau* torch.tanh(net)
# MAML Training Loop

def main():
    device = torch.device("cpu")
    
    # Hyperparameters
    learning_rate = 1e-3
    input_dim = 3
    output_dim = 2
    hidden_dim = 8
    num_layers = 1

    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    residual_mlp = MLP(input_dim = input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers) # the network
    start = time.time()

    for param in residual_mlp.parameters():
        param.requires_grad = False
    residual_mlp = residual_mlp
    residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=learning_rate, weight_decay=0.1) # lr = learning rate, the optimizer
    #residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=learning_rate) # lr = learning rate, the optimizer
    
    residual_criterion = nn.MSELoss()
    X_batch = torch.tensor(X_train, dtype=torch.float32)
    y_target = torch.tensor(Y_target, dtype=torch.float32)
    
    for p in residual_mlp.parameters(): p.requires_grad = True
    for _ in range(200):
        residual_optimizer.zero_grad() # optimizer object
        prediction = residual_mlp(X_batch) # gives data to network to make a prediction
        loss = residual_criterion(prediction, y_target)
        #l1_norm = sum(torch.linalg.norm(p, 1) for p in residual_mlp.parameters())
        #l2_norm = sum(p.pow(2).sum() for p in self.residual_mlp.parameters())
        #regularization = 0.1
        #loss += regularization * l1_norm
        loss.backward() # calculates gradient
        residual_optimizer.step() # one optimization step to update parameters
    end = time.time()
    torch.save(residual_mlp.state_dict(), "cascaded_tanks_pretrain.pth")

    print("elapsed", 1000*(end-start))

if __name__ == "__main__":
    main()
    

