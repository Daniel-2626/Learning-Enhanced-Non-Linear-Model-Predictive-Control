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
#csv_path = os.path.join(os.path.dirname(__file__), 'residuals_mismatched_heating.csv')
#df = pd.read_csv(csv_path)
#print(df.head())

class MLP(nn.Module):

    def __init__(self, input_dim=4, output_dim=2, hidden_dim=16, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        self.tau = nn.Parameter(torch.tensor([0.001]), requires_grad=False)        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):

        net = self.net(x)
        #return net
        return self.tau * torch.tanh(net) # Kinda BS cuz then our residual NN is basically just a tanh function
        # Or in other words: An expected value in two parts
        # But not really. Not necssarily tanh in time, just in inputs.

    # Hyperparameters
learning_rate = 1e-2
input_dim = 4
output_dim = 1
hidden_dim = 32
num_layers = 2
device = torch.device("cpu")

model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
residual_mlp = MLP(input_dim = input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers) # the network
residual_mlp.load_state_dict(torch.load("simulation_network.pth", weights_only=True))

def plot_control_jacobians_wrt_q(model, T_bat_fixed, current_fixed, omega_scale, Q_heat_scale):
    model.eval() # Set to evaluation mode
    
    # Create a range for Q_heat (normalized -1 to 1)
    q_range = torch.linspace(-4, 4, 100).reshape(-1, 1)
    # Fixed inputs (normalized)
    t_in = torch.full_like(q_range, T_bat_fixed / 100.0)
    i_in = torch.full_like(q_range, current_fixed / 25.0)
    w_in = torch.full_like(q_range, 2) # Fixed pump at 50%
    
    # Concatenate: [T, I, Omega, Q]
    inputs = torch.cat([t_in, i_in, w_in, q_range], dim=1)
    inputs.requires_grad_(True)
    
    # Forward pass
    residual = model(inputs)
    
    # Calculate Jacobian with respect to inputs
    # We want d(residual)/d(input)
    grads = torch.autograd.grad(
        outputs=residual,
        inputs=inputs,
        grad_outputs=torch.ones_like(residual),
        create_graph=False
    )[0]
    
    # Extract gradients for Q_heat (index 3) and Omega (index 2)
    d_res_dq = grads[:, 3].detach().numpy()
    d_res_dw = grads[:, 2].detach().numpy()
    res_val = residual.detach().numpy()

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Normalized Q_heat')
    ax1.set_ylabel('Residual Value', color='blue')
    ax1.plot(q_range.numpy(), res_val, label='Residual Value', color='blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Jacobian (Sensitivity)', color='red')
    ax2.plot(q_range.numpy(), d_res_dq, label='d(Res)/d(Q)', color='red', linestyle='--')
    ax2.axhline(0, color='black', lw=1)
    
    plt.title(f'MLP Sensitivity Analysis (T={T_bat_fixed}°C, I={current_fixed}A)')
    fig.tight_layout()
    plt.show()

    return q_range.numpy(), res_val, d_res_dq

def plot_control_jacobians_wrt_w(model, T_bat_fixed, current_fixed, omega_scale, Q_heat_scale):
    model.eval() # Set to evaluation mode
    
    # Create a range for Q_heat (normalized -1 to 1)
    w_range = torch.linspace(0.15, 4.18, 100).reshape(-1, 1)
    # Fixed inputs (normalized)
    t_in = torch.full_like(w_range, T_bat_fixed / 100.0)
    i_in = torch.full_like(w_range, current_fixed / 25.0)
    q_in = torch.full_like(w_range, 2) # Fixed Q heat at 2000 W
    
    # Concatenate: [T, I, Omega, Q]
    inputs = torch.cat([t_in, i_in, w_range, q_in], dim=1)
    inputs.requires_grad_(True)
    
    # Forward pass
    residual = model(inputs)
    
    # Calculate Jacobian with respect to inputs
    # We want d(residual)/d(input)
    grads = torch.autograd.grad(
        outputs=residual,
        inputs=inputs,
        grad_outputs=torch.ones_like(residual),
        create_graph=False
    )[0]
    
    # Extract gradients for Q_heat (index 3) and Omega (index 2)
    #d_res_dq = grads[:, 3].detach().numpy()
    d_res_dw = grads[:, 2].detach().numpy()
    res_val = residual.detach().numpy()

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Normalized w')
    ax1.set_ylabel('Residual Value', color='blue')
    ax1.plot(w_range.numpy(), res_val, label='Residual Value', color='blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Jacobian (Sensitivity)', color='red')
    ax2.plot(w_range.numpy(), d_res_dw, label='d(Res)/d(w)', color='red', linestyle='--')
    ax2.axhline(0, color='black', lw=1)
    
    plt.title(f'MLP Sensitivity Analysis (T={T_bat_fixed}°C, I={current_fixed}A)')
    fig.tight_layout()
    plt.show()

    return w_range.numpy(), res_val, d_res_dw

def plot_control_jacobians_wrt_i(model, T_bat_fixed, current_fixed, omega_scale, Q_heat_scale):
    model.eval() # Set to evaluation mode
    
    # Create a range for Q_heat (normalized -1 to 1)
    i_range = torch.linspace(0, 80/25, 100).reshape(-1, 1)
    # Fixed inputs (normalized)
    t_in = torch.full_like(i_range, T_bat_fixed / 100.0)
    #i_in = torch.full_like(w_range, current_fixed / 25.0)
    
    q_in = torch.full_like(i_range, 2) # Fixed Q heat at 2000 W
    w_in = torch.full_like(i_range, 2) # Fixed pump at 50%

    # Concatenate: [T, I, Omega, Q]
    inputs = torch.cat([t_in, i_range, w_in, q_in], dim=1)
    inputs.requires_grad_(True)
    
    # Forward pass
    residual = model(inputs)
    
    # Calculate Jacobian with respect to inputs
    # We want d(residual)/d(input)
    grads = torch.autograd.grad(
        outputs=residual,
        inputs=inputs,
        grad_outputs=torch.ones_like(residual),
        create_graph=False
    )[0]
    
    # Extract gradients for Q_heat (index 3) and Omega (index 2)
    #d_res_dq = grads[:, 3].detach().numpy()
    d_res_di = grads[:, 1].detach().numpy()
    res_val = residual.detach().numpy()

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Normalized i')
    ax1.set_ylabel('Residual Value', color='blue')
    ax1.plot(i_range.numpy(), res_val, label='Residual Value', color='blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Jacobian (Sensitivity)', color='red')
    ax2.plot(i_range.numpy(), d_res_di, label='d(Res)/d(i)', color='red', linestyle='--')
    ax2.axhline(0, color='black', lw=1)
    
    plt.title(f'MLP Sensitivity Analysis (T={T_bat_fixed}°C, I={current_fixed}A)')
    fig.tight_layout()
    plt.show()

    return i_range.numpy(), res_val, d_res_di

def plot_control_jacobians_wrt_T(model, T_bat_fixed, current_fixed, omega_scale, Q_heat_scale):
    model.eval() # Set to evaluation mode
    
    # Create a range for Q_heat (normalized -1 to 1)
    # Fixed inputs (normalized)

    t_range = torch.linspace(263.15/100, 303.15/100, 100).reshape(-1, 1)
    #i_in = torch.full_like(w_range, current_fixed / 25.0)
    i_in = torch.full_like(t_range, current_fixed / 25.0)

    q_in = torch.full_like(t_range, 2) # Fixed Q heat at 2000 W
    w_in = torch.full_like(t_range, 2) # Fixed pump at 50%

    # Concatenate: [T, I, Omega, Q]
    inputs = torch.cat([t_range, i_in, w_in, q_in], dim=1)
    inputs.requires_grad_(True)
    
    # Forward pass
    residual = model(inputs)
    
    # Calculate Jacobian with respect to inputs
    # We want d(residual)/d(input)
    grads = torch.autograd.grad(
        outputs=residual,
        inputs=inputs,
        grad_outputs=torch.ones_like(residual),
        create_graph=False
    )[0]
    
    # Extract gradients for Q_heat (index 3) and Omega (index 2)
    #d_res_dq = grads[:, 3].detach().numpy()
    d_res_dT = grads[:, 0].detach().numpy()
    res_val = residual.detach().numpy()

    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Normalized T')
    ax1.set_ylabel('Residual Value', color='blue')
    ax1.plot(t_range.numpy(), res_val, label='Residual Value', color='blue', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Jacobian (Sensitivity)', color='red')
    ax2.plot(t_range.numpy(), d_res_dT, label='d(Res)/d(T)', color='red', linestyle='--')
    ax2.axhline(0, color='black', lw=1)
    
    plt.title(f'MLP Sensitivity Analysis (T={T_bat_fixed}°C, I={current_fixed}A)')
    fig.tight_layout()
    plt.show()

    return t_range.numpy(), res_val, d_res_dT


q_range, res_val_q, d_res_dq  = plot_control_jacobians_wrt_q(residual_mlp, 293.15, 40, 100, 1000)
w_range, res_val_w, d_res_dw  = plot_control_jacobians_wrt_w(residual_mlp, 293.15, 40, 100, 1000)
i_range, res_val_i, d_res_di = plot_control_jacobians_wrt_i(residual_mlp, 293.15, 40, 100, 1000)
T_range, res_val_T, d_res_dT  = plot_control_jacobians_wrt_T(residual_mlp, 293.15, 40, 100, 1000)
q_range = q_range.flatten()
res_val_q = res_val_q.flatten()
d_res_dq = d_res_dq.flatten()

w_range = w_range.flatten()
res_val_w = res_val_w.flatten()
d_res_dw = d_res_dw.flatten()

i_range = i_range.flatten()
res_val_i = res_val_i.flatten()
d_res_di = d_res_di.flatten()

T_range = T_range.flatten()
res_val_T = res_val_T.flatten()
d_res_dT = d_res_dT.flatten()
print("q_range", q_range.shape, res_val_q.shape, d_res_dq.shape)

data = {"q_range": q_range, "res_val_q": res_val_q, "d_res_dq": d_res_dq, 
"w_range": w_range, "res_val_w": res_val_w, "d_res_dw": d_res_dw, 
"i_range": i_range, "res_val_i": res_val_i, "d_res_di": d_res_di, 
"T_range": T_range, "res_val_T": res_val_T, "d_res_dT": d_res_dT}

jac_df = pd.DataFrame(data=data)
jac_df.to_csv("jacobian.csv", index=False)

