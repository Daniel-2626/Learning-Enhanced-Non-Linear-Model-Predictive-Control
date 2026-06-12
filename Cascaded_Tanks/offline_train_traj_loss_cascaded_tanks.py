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
import casadi as cs

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Load dataset of input features (h1, h2 and u)
csv_path = os.path.join(os.path.dirname(__file__), 'cascaded_nominal_matched.csv')
df = pd.read_csv(csv_path)
inputs = df[['h1','h2','u']].to_numpy()
X_train = inputs

# Control model
A1 = 1
a1 = 0.1
A2 = 1
a2 = 0.1
rho = 1000
k = 1000
g = 9.82

h1 = cs.SX.sym('h1')
h2 = cs.SX.sym('h2') 
u = cs.SX.sym('u_in')

h1_dot = k*u/(rho*A1) - a1/A1 * cs.sqrt(2*g*h1+0.00001)
h2_dot = a1/A1 * cs.sqrt(2*g*h1 + 0.00001) - a2/A2 * cs.sqrt(2*g*h2 + 0.00001)

ode = cs.vertcat(h1_dot, h2_dot)
states = cs.vertcat(
    h1,
    h2
)
f = cs.Function("f", [states,u], [ode], ["x","u"], ["ode"])
"""
Since Cascaded Tanks use RK4 you need to do work to properly do trajectory sim using it
# In Acados, we will integrate f_nom + f_res using RK4
# Since RK4 is nonlinear, this is NOT the same as RK4(f_nom) + RK4(f_res)
Define help functions for performing this
"""
def RK4(state, input_u, dt, f):
    # Used to simulate the state when running MPC
    # Only used here for evaluation of NN
    K1 = f(state, input_u)
    K2 = f(state + dt/2 * K1, input_u)
    K3 = f(state + dt/2 * K2, input_u)
    K4 = f(state + dt * K3, input_u)
    next_state = state + (dt/6) * (K1 + 2*K2 + 2*K3 +K4)
    return next_state

nominal_ratio = 1
# Convert nominal dynamics to a torch formulation 
def nominal_dynamics_torch(state, u):
    # Match the nominal parameters from CasADi model
    # I.e. these should be the SAME dynamics as inside of the MPC
    A1 = 1 
    a1 = 0.1*nominal_ratio 
    A2 = 1 
    a2 = 0.1*nominal_ratio
        
    k = 1000
    rho = 1000 
        
    g = 9.82
    
    # Dynamics
        
    h1 = state[0, 0]
    h2 = state[1, 0]
    
    h1_dot = k * u / (rho * A1) - (a1 / A1) * torch.sqrt(2 * g * h1 + 0.00001)
    h2_dot = (a1 / A1) * torch.sqrt(2 * g * h1 + 0.00001) - (a2 / A2) * torch.sqrt(2 * g * h2 + 0.00001)
    
    return torch.cat([h1_dot.reshape(1,1), h2_dot.reshape(1,1)], dim=0)

def RK4_nominal_torch(state, u, dt):
    K1 = nominal_dynamics_torch(state, u)
    K2 = nominal_dynamics_torch(state + dt/2 * K1, u)
    K3 = nominal_dynamics_torch(state + dt/2 * K2, u)
    K4 = nominal_dynamics_torch(state + dt * K3, u)
    return state + (dt/6) * (K1 + 2*K2 + 2*K3 + K4)

def combined_dynamics_torch(state, u, residual_model):
    # 1. Get nominal continuous derivative (2x1 vector)
    dot_nom = nominal_dynamics_torch(state, u)
    
    # 2. Get residual continuous derivative
    # residual_mlp expects (1x2) for x and (1x1) for u
    x_in = state.reshape(1, 2)
    u_in = u.reshape(1, 1)
    dot_res = residual_model(x_in, u_in).reshape(2, 1)
    
    # 3. Combine them BEFORE integration (matches Acados f_expl)
    return dot_nom + dot_res

def RK4_combined_torch(state, u, dt, residual_model):
    K1 = combined_dynamics_torch(state, u, residual_model)
    K2 = combined_dynamics_torch(state + dt/2 * K1, u, residual_model)
    K3 = combined_dynamics_torch(state + dt/2 * K2, u, residual_model)
    K4 = combined_dynamics_torch(state + dt * K3, u, residual_model)
    return state + (dt/6) * (K1 + 2*K2 + 2*K3 + K4)
### MLP ARCHITECTURE ###
# input_dim, output_dim, hidden_dim, num_layers will be changed lower in the code
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=3):
        super(MLP, self).__init__()
        self.tau = 0.2
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()] # nn.Tanh hyperbolic tangent
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]) # nn.Linear applies an affine transform
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, u):
        # RK4_combined_torch and L4CasADi expect different format on inputs x, u
        # For RK4 it is more convenient to have them be different (since in RK4 we propagate forward state, but input is constant)
        # L4CasADi seemingly expects them to be the same argument
        # This ensures that the forward argument accepts them as different arguments but then concatanate them so it aligns with L4CasADi
        inp = torch.cat([x,u], dim=1)
        net =  self.net(inp)
        # Bound NN output by tanh and confidence parameter tau
        return self.tau* torch.tanh(net)


# Training 
def main():
    device = torch.device("cpu")
    
    # Hyperparameters (Must have the same input_dim, output_dim, hidden_dim and num_layers as being used in the adaptive)

    learning_rate = 1e-2
    input_dim = 3
    output_dim = 2
    hidden_dim = 8
    num_layers = 2

    residual_mlp = MLP(input_dim = input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers) # the network
    start = time.time()

    residual_mlp = residual_mlp
    residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=learning_rate, weight_decay=0.01) # lr = learning rate, the optimizer

    data = df[['h1', 'h2', 'u']].to_numpy()

    print(data)

    start = time.time()

    X_batch = torch.tensor(data, dtype=torch.float32)
    print(X_batch)
    
    N = 50 # Simulation steps (here chosen the same as MPC horizon)
    T_rollout = N 

    dt = 0.5
    dt_torch = torch.tensor(dt, dtype=torch.float32)
    n_rows = len(X_batch)
    print(n_rows)
    # Epoch loop
    T_rollout = 10
    weight = 1.0  # Optional discount factor if you want to weight steps differently

    for param in residual_mlp.parameters():
        param.requires_grad = True

    for epoch in range(100):
        residual_optimizer.zero_grad()
        loss = 0
        num_predictions = 0
            
        # Loop through the batch, jumping by the rollout length
        for k0 in range(0, N, T_rollout):
            # Always start the rollout window from the true measured state
            h1 = X_batch[k0, 0]
            h2 = X_batch[k0, 1]
            h_pred = torch.cat([h1.reshape(1,1), h2.reshape(1,1)], dim=0) # 2x1 vector
                
            # Rollout sequentially for T_rollout steps
            for step in range(T_rollout):
                t = k0 + step
                    
                if (t + 1) >= n_rows:
                    break
                        
                u_in = X_batch[t, 2].reshape(1,1)
                h_next_true = torch.cat([X_batch[t+1, 0].reshape(1,1), X_batch[t+1, 1].reshape(1,1)], dim=0)

                    
                # Calculate jacobian
                mlp_input_jac = h_pred.detach().reshape(1, 2).requires_grad_(True)
                dot_res_jac = residual_mlp(mlp_input_jac, u_in)
                # Jacobian wrt h1
                grad_f1 = torch.autograd.grad(
                    outputs=dot_res_jac[0, 0], inputs=mlp_input_jac, 
                    grad_outputs=torch.ones_like(dot_res_jac[0, 0]),
                    create_graph=True, retain_graph=True
                )[0]
                # Jacobian wrt h2
                grad_f2 = torch.autograd.grad(
                    outputs=dot_res_jac[0, 1], inputs=mlp_input_jac, 
                    grad_outputs=torch.ones_like(dot_res_jac[0, 1]),
                    create_graph=True, retain_graph=True
                )[0]
                    
                weight_jac = 0.01 
                loss_jac = weight_jac * (grad_f1[0, 0]**2 + grad_f2[0, 0]**2 + 
                                        grad_f1[0, 1]**2 + grad_f2[0, 1]**2)


                h_pred_next = RK4_combined_torch(h_pred, u_in, dt, residual_mlp)
                    
                # Base Mean Squared Error for tracking the true state
                loss_mse = ((h_next_true - h_pred_next) ** 2).mean()
                    
                # Accumulate all losses for this rollout step
                loss += (weight**step) * (loss_mse + loss_jac)
                num_predictions += 1
                    
                h_pred = h_pred_next            
        loss.backward()
        residual_optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Training loss: {loss.item():.6f}")
                

    end = time.time()
    # Saving trained NN parameters
    torch.save(residual_mlp.state_dict(), "cascaded_tanks_traj_pretrain.pth")
    
    ### Plotting (comment if not wanted)

    h1 = X_batch[0,0]
    h2 = X_batch[0,1]
               
    h = torch.cat([h1.reshape(1,1),h2.reshape(1,1)], dim=1).reshape((2,1))

    h1_num = data[0,0]
    h2_num = data[0,1]
    h_num = np.array(h1_num,h2_num)
    h_preds = []
    h_preds.append(h.detach().numpy().flatten())

    # Simulate using adaptive control model
    for k in range(0,N):

        h_num = h.detach().numpy()
   
        u = X_batch[k,2]       
        u_num = data[k,2]     
        
                
               
        mlp_input = torch.cat([
            (h[0]).reshape(1,1),
            (h[1]).reshape(1,1),
                ], dim=1).float()
        mlp_input.requires_grad_(True)
        #residual = residual_mlp(mlp_input, u.reshape(1,1))
        #dh = torch.from_numpy(RK4(h_num,u_num,dt,f).full())
     
        u_in = u.reshape((1,1)).float()
        
        h_pred = RK4_combined_torch(h, u, dt, residual_mlp)
        
        h_preds.append(h_pred.detach().numpy().flatten())
        h = h_pred

    h_nom_preds = []

    h1_num = data[0,0]
    h2_num = data[0,1]
    h_num = np.array([h1_num,h2_num])
    h_nom_preds.append(h_num)
    print("h_num", h_num)
    # Simulate using nominal control model
    for k in range(0,N):    
        u_num = data[k,2]   
        print(u_num)  
        
        dh = RK4(h_num,u_num,dt,f).full()

        h_pred = dh 
        h_nom_preds.append(h_pred.flatten())
        h_num = h_pred
    print("h_pred.flatten()", h_pred.flatten())
    h_preds = np.array(h_preds)
    h_nom_preds = np.array(h_nom_preds)

    plt.plot(h_preds[:-1,0])
    plt.plot(h_nom_preds[:-1,0])
    plt.plot(data[:N,0])
    plt.legend(["h_nn","hnom", "htrue"])
    plt.show()



    print("elapsed", 1000*(end-start))

if __name__ == "__main__":
    main()
    

