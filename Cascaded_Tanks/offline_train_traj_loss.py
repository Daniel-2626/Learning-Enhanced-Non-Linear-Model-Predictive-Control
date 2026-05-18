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
# Load dataset of residuals
csv_path = os.path.join(os.path.dirname(__file__), 'cascaded_nominal_mismatched.csv')
df = pd.read_csv(csv_path)
print(df.head())

inputs = df[['h1','h2','u']].to_numpy()
X_train = inputs

#X_train = np.transpose(X_train)
print(X_train.shape)

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
## Actual model
h1_dot = k*u/(rho*A1) - a1/A1 * cs.sqrt(2*g*h1+0.00001) #+ a2/A2 * cs.sqrt(2*g*h2 + 0.00001) 
#h1_dot = k*u/(rho*A1)*cs.exp(-u/10) - a1/A1 * cs.sqrt(2*g*h1+0.00001) #+ a2/A2 * cs.sqrt(2*g*h2 + 0.00001) 

h2_dot = a1/A1 * cs.sqrt(2*g*h1 + 0.00001) - a2/A2 * cs.sqrt(2*g*h2 + 0.00001) #- 0.1*k*u/(rho*A2)
ode = cs.vertcat(h1_dot, h2_dot)
states = cs.vertcat(
    h1,
    h2
)
f = cs.Function("f", [states,u], [ode], ["x","u"], ["ode"])

def RK4(state, input_u, dt, f):
    K1 = f(state, input_u)
    K2 = f(state + dt/2 * K1, input_u)
    K3 = f(state + dt/2 * K2, input_u)
    K4 = f(state + dt * K3, input_u)
    next_state =  (dt/6) * (K1 + 2*K2 + 2*K3 +K4)
    return next_state

def RK4_torch(model, state, inp, dt):

    K1 = model(state, inp)
    K2 = model(state + dt/2 *K1, inp)
    K3 = model(state + dt/2 * K2,inp)
    K4 = model(state + dt * K3, inp)

    return dt/6 *(K1+2*K2 + 2*K3 + K4)
#print(Y_target)
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=3):
        super(MLP, self).__init__()
        self.tau = 2
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()] # nn.ReLU rectified linear function (max(x,0))
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]) # nn.Linear applies an affine transform. hidden_dim features and hidden_dim out features
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x, u):
        inp = torch.cat([x,u], dim=1)
        net =  self.net(inp)
        #return net
        return self.tau* torch.tanh(net)
# MAML Training Loop

def main():
    device = torch.device("cpu")
    
    # Hyperparameters
    learning_rate = 1e-3
    input_dim = 3
    output_dim = 2
    hidden_dim = 8
    num_layers = 2

    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    residual_mlp = MLP(input_dim = input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers) # the network
    start = time.time()

    residual_mlp = residual_mlp
    residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=learning_rate, weight_decay=0.01) # lr = learning rate, the optimizer

    h1_true = df[['h1']].to_numpy().flatten()
    h2_true = df[['h2']].to_numpy().flatten()
    u_true = df[['u']].to_numpy().flatten()


    start = time.time()


    n_rows = len(u_true)

    h1_true_tensor = torch.tensor(h1_true, dtype=torch.float32)
    h2_true_tensor = torch.tensor(h2_true,  dtype=torch.float32)
    u_true_tensor = torch.tensor(u_true, dtype=torch.float32)
    #lambda_nn =0.1
    T_rollout = 1

    dt = 0.5
    dt_torch = torch.tensor(dt, dtype=torch.float32)
    
    for epoch in range(500):
        residual_optimizer.zero_grad() # optimizer object
        loss = 0
        residual_loss = 0
        weight = 1#1.02
        # We'll rollout for 10 steps and then update T
        h1 = h1_true_tensor[0]
        h2 = h2_true_tensor[0]
               
        h = torch.cat([h1.reshape(1,1),h2.reshape(1,1)], dim=1).reshape((2,1))

        h1_num = h1_true[0]
        h2_num = h2_true[0]
        h_num = np.array([h1_num,h2_num])
        N = 50
        for k0 in range(0,N, T_rollout):
            #print(k0)
            h1_num = h1_true[k0]
            h2_num = h2_true[k0]
            h_num = np.array([h1_num,h2_num])
            h1 = h1_true_tensor[k0]
            h2 = h2_true_tensor[k0]
            h = torch.cat([h1.reshape(1,1),h2.reshape(1,1)], dim=1).reshape((2,1))

            for i in range(T_rollout):

                t = k0 + i
                if (t+1) >= n_rows:
                    break
           
                u = u_true_tensor[t]       
                u_num = u_true[t]     
                h1_next = h1_true_tensor[t+1]
                h2_next = h2_true_tensor[t+1]
                h_next = torch.cat([h1_next.reshape(1,1),h2_next.reshape(1,1)], dim=1).reshape((2,1))

                
               
                mlp_input = torch.cat([
                    (h[0]).reshape(1,1),
                    (h[1]).reshape(1,1),
                ], dim=1).float()
                #print(mlp_input.shape)
                #mlp_input = h.reshape((1,2))
                #print("mlp input shape", mlp_input.shape)
                #print(i)
                mlp_input.requires_grad_(True)
                #print("type mlp input", type(mlp_input))
                #print(mlp_input)
                #print("type u", type(u))
                #print(u.reshape((1,1)))
                u_in = u.reshape(1,1).float()
                residual = residual_mlp(mlp_input,u_in)

                dh = torch.from_numpy(RK4(h_num,u_num,dt,f).full()).float()
     
                #print("dh shape", dh.shape)

                #mlp_input.requires_grad_(True)
                dh_MLP = RK4_torch(residual_mlp, mlp_input, u_in, dt).reshape((2,1))
                #print("dh_MLP shape", dh_MLP.shape)
                h_pred = h + dh + dh_MLP

                loss +=  weight**i * ((h_next - h_pred)**2).mean()
                h = h_pred
        
     
            h = h.detach().clone()
        
        loss.backward() # calculates gradient

    
        residual_optimizer.step() # one optimization step to update parameters

        if epoch % 20 == 0:
            print("Epoch {}: Training loss: {}".format(epoch, loss.item()))

    end = time.time()

    h1 = h1_true_tensor[0]
    h2 = h2_true_tensor[0]
               
    h = torch.cat([h1.reshape(1,1),h2.reshape(1,1)], dim=1).reshape((2,1))

    h1_num = h1_true[0]
    h2_num = h2_true[0]
    h_num = np.array(h1_num,h2_num)
    print(h)
    h_preds = []
    h_preds.append(h.detach().numpy().flatten())

    #h_preds.append(h.detach().numpy())
    for k in range(0,N):
        #h = torch.cat([h1.reshape(1,1),h2.reshape(1,1)], dim=1).reshape((2,1))
        #print(k0)
        h_num = h.detach().numpy()
   
        u = u_true_tensor[k]       
        u_num = u_true[k]     
        
                
               
        mlp_input = torch.cat([
            (h[0]).reshape(1,1),
            (h[1]).reshape(1,1),
                ], dim=1).float()
        mlp_input.requires_grad_(True)
        residual = residual_mlp(mlp_input, u.reshape(1,1))
        dh = torch.from_numpy(RK4(h_num,u_num,dt,f).full())
     
        u_in = u.reshape((1,1)).float()
        
        dh_MLP = RK4_torch(residual_mlp, mlp_input, u_in, dt).reshape((2,1))
        #dh = RK4(h_num,u_num,dt,f).full()
        dh = torch.from_numpy(RK4(h_num,u_num,dt,f).full()).float()

        h_pred = h + dh + dh_MLP
        #h_preds.append(h_pred.flatten())
        h_preds.append(h_pred.detach().numpy().flatten())
        #print(h_pred)
        #h_num = h_pred
        h = h_pred

    h_nom_preds = []

    h1_num = h1_true[0]
    h2_num = h2_true[0]
    h_num = np.array(h1_num,h2_num)
    for k in range(0,N):
        #h = torch.cat([h1.reshape(1,1),h2.reshape(1,1)], dim=1).reshape((2,1))
        #print(k0)
        
        #u = u_true_tensor[k]       
        u_num = u_true[k]     
        
                
               
        #mlp_input = torch.cat([
        #    (h[0]).reshape(1,1),
        #    (h[1]).reshape(1,1),
        #        ], dim=1).float()
        #mlp_input.requires_grad_(True)
        #residual = residual_mlp(mlp_input, u.reshape(1,1))
        dh = RK4(h_num,u_num,dt,f).full()
     
        #u_in = u.reshape((1,1)).float()
        
        #dh_MLP = RK4_torch(residual_mlp, mlp_input, u_in, dt).reshape((2,1))
        #dh = RK4(h_num,u_num,dt,f).full()
        dh = RK4(h_num,u_num,dt,f).full()

        h_pred = h_num + dh #+ dh_MLP
        h_nom_preds.append(h_pred.flatten())
        #h_preds.append(h_pred)
        #print(h_pred)
        #h_num = h_pred
        h_num = h_pred
    #print(h_preds)

    h_preds = np.array(h_preds)
    h_nom_preds = np.array(h_nom_preds)
    print(h_preds)
    print(h_preds.shape)
    plt.plot(h_preds[:,0])
    plt.plot(h_nom_preds[:,0])
    plt.plot(h1_true[:N])
    plt.legend(["hpred","hnom", "htrue"])
    plt.show()
    torch.save(residual_mlp.state_dict(), "cascaded_tanks_traj_pretrain.pth")

    print("elapsed", 1000*(end-start))

if __name__ == "__main__":
    main()
    

