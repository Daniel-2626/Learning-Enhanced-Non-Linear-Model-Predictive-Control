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
"""
Offline training works by reading a csv of the data 
Each input and output should be in a named column
"""
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# Load dataset of residuals
csv_path = os.path.join(os.path.dirname(__file__), 'Training_Data/residuals_matched_heating.csv')
df = pd.read_csv(csv_path)
print(df.head())

class MLP(nn.Module):

    def __init__(self, input_dim=4, output_dim=2, hidden_dim=16, num_layers=2):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        # Confidence parameter in NN
        self.tau = nn.Parameter(torch.tensor([0.001]), requires_grad=False)        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):

        net = self.net(x)
        # bound NN by hyperbolic tangent
        return self.tau * torch.tanh(net) 

# Training loop
def main():
    device = torch.device("cpu")
    start_idx = 0
    end_idx = 200
    CELSIUS_TO_KELVIN = 273.15
    omega_scale = 100
    Q_heat_scale = 1000
    T_bat_scale = 100
    current_scale = 25


    polyorder = 2
    window_length = 10
    # Filtering
    temperatures_data = df[['T_bat']].to_numpy().flatten()[start_idx:end_idx]

    temperatures_filtered = savgol_filter(temperatures_data, window_length=window_length, polyorder=polyorder)

    current_data = df[['current']].to_numpy().flatten()[start_idx:end_idx]
    current_filtered = savgol_filter(current_data, window_length=window_length, polyorder=polyorder)

    omega_scaled_data = df[['omega_scaled']].to_numpy().flatten()[start_idx:end_idx]
    omega_scaled_filtered = savgol_filter(omega_scaled_data, window_length=window_length, polyorder=polyorder)

    Q_heat_scaled_data = df[['Q_heat_scaled']].to_numpy().flatten()[start_idx:end_idx]
    Q_heat_scaled_filtered = savgol_filter(Q_heat_scaled_data, window_length=window_length, polyorder=polyorder)


    # Hyperparameters
    learning_rate = 1e-2
    input_dim = 4
    output_dim = 1
    hidden_dim = 32
    num_layers = 2
    weight_decay=0.1
    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    residual_mlp = MLP(input_dim = input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers) # the network
    start = time.time()

    for param in residual_mlp.parameters():
        param.requires_grad = False
    residual_mlp = residual_mlp
    residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay) # lr = learning rate, the optimizer    

    # Calculate derivative with model
    # Parameters
    m_battery = 20*2.5*4
    c_battery = 795
    c_coolant = 3500
    density_coolant = 1050
                    
    pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
    R_battery = 4*20*0.0128 # Battery resistance
    C_battery = 28*3600 # in coloumb
    hA_bat = 2500
    alpha_0 = 0.635039 
    alpha_1 = 0.915692 
    alpha_2 = 0.919681  
    alpha_3 = 1.47275
    gamma = 7.38325

    dt = 5
    T_env = (temperatures_data[0]*T_bat_scale)
    print("T_env", T_env)

    n_rows = end_idx - start_idx
    dynamics_model = []
    for k in range(n_rows):
        omega_scaled = omega_scaled_filtered[k]
        omega = omega_scale*omega_scaled
        Q_heat_scaled = Q_heat_scaled_filtered[k]
        Q_heat = Q_heat_scale*Q_heat_scaled
                            
        current_scaled = current_filtered[k]            
        current = current_scale*current_scaled
                
        T_bat = T_bat_scale* temperatures_filtered[k]
        # Dynamics
        mdot_c = density_coolant*pump_displacement*omega

        NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
        T_clin= (T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
        T_clout = ((T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat)
                 
        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))
        dynamics_model.append(T_bat_dot_model)
    # The savgol derivative is assumed to be true dynamics
    dynamics_savgol = T_bat_scale*savgol_filter(temperatures_data, window_length=window_length, polyorder=polyorder, deriv = 1, delta = dt)

    confidence = 0.001
    # For automatic update of confidence. This implementation did not perform well
    #confidence = np.sqrt(np.power(dynamics_savgol-dynamics_model,2).mean())
    #confidence = np.clip(confidence, 0.001, 0.05)
    with torch.no_grad():
        residual_mlp.tau.copy_(torch.tensor([confidence]))
    print("confidence", confidence)
    # Unfreeze parameters for training
    for p in residual_mlp.parameters(): p.requires_grad = True


    temperatures_tensor = torch.tensor(temperatures_filtered, dtype=torch.float32)
    current_tensor = torch.tensor(current_filtered,  dtype=torch.float32)
    omega_scaled_tensor = torch.tensor(omega_scaled_filtered, dtype=torch.float32)
    Q_heat_scaled_tensor = torch.tensor(Q_heat_scaled_filtered, dtype=torch.float32)
    regularization = 0

    T_rollout = 20
    lambda_nn = 1
    lambda_phys = 1
    lambda_jac = 0.1
    alpha = 1
    epsilon = 0.1
    noise_std = 0.005
    # Epoch loop
    for epoch in range(200):
        residual_optimizer.zero_grad() # optimizer object
        loss = 0
        T_bat = T_bat_scale*temperatures_tensor[0]
        weight = 1.02
        # Rollout for 20 steps (i.e. simulate for the next 20 steps, then re-initialize)
        for k0 in range(0,n_rows-1, T_rollout):
            T_bat =  T_bat_scale*temperatures_tensor[k0]

            for i in range(T_rollout):

                t = k0 + i
                if (t+1) >= n_rows:
                    break
                omega_scaled = omega_scaled_tensor[t]
                omega = omega_scale*omega_scaled
                Q_heat_scaled = Q_heat_scaled_tensor[t]
                Q_heat = Q_heat_scale*Q_heat_scaled
                            
                current_scaled = current_tensor[t]            
                current = current_scale*current_scaled
                
                T_bat_next = T_bat_scale* temperatures_tensor[t+1]
                T_bat_noisy = T_bat + torch.randn_like(T_bat) * noise_std  # Add noise to teach NN not to overreact

                # Dynamics
                mdot_c = density_coolant*pump_displacement*omega

                NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
                T_clin= (T_bat_noisy + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
                T_clout = ((T_clin - T_bat_noisy) * alpha_2*np.exp(-NTU_bat) + T_bat_noisy)
                    
                Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

                T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat_noisy))
                
              
                mlp_input = torch.cat([
                    (T_bat_noisy/T_bat_scale).reshape(1,1),
                    (current/current_scale).reshape(1,1),
                    (omega/omega_scale).reshape(1,1),
                    (Q_heat/Q_heat_scale).reshape(1,1)
                ], dim=1)
                mlp_input.requires_grad_(True)
                residual = residual_mlp(mlp_input)
                jacobian = torch.autograd.grad(
                outputs=residual,
                inputs=mlp_input,
                grad_outputs=torch.ones_like(residual),
                create_graph=True,
                retain_graph=True
                 )[0]

                jacobian_T = jacobian[:, 0] # Jacobian wrt state (temperature)


                jacobian_penalty = jacobian_T.pow(2).mean()


                T_bat_pred = T_bat_noisy + dt*(T_bat_dot_model+ lambda_nn*residual)
                error = T_bat_next - T_bat_pred 
                loss +=  weight**i * torch.where(torch.abs(error)< epsilon, torch.zeros_like(error), error**2) 
                loss += lambda_jac*jacobian_penalty

                T_bat = T_bat_pred.squeeze()
            T_bat = T_bat.detach().clone()

        loss.backward() # calculates gradient
        nn.utils.clip_grad_norm_(
            residual_mlp.parameters(),
            max_norm= 0.1
        )    
        residual_optimizer.step() # one optimization step to update parameters

        if epoch % 20 == 0:
            print("Epoch {}: Training loss: {}".format(epoch, loss.item()))
    end = time.time()
    print("elapsed", 1000*(end-start))

    # Freeze for evaluation of NN
    for p in residual_mlp.parameters(): 
        p.requires_grad = False

    # Evaluation of NN (comment if not needed)
    polyorder = 2
    window_length = 10
    start_idx = 0
    end_idx = 200
    temperatures_data = df[['T_bat']].to_numpy().flatten()[start_idx:end_idx]

    temperatures_filtered = savgol_filter(temperatures_data, window_length=window_length, polyorder=polyorder)

    current_data = df[['current']].to_numpy().flatten()[start_idx:end_idx]
    current_filtered = savgol_filter(current_data, window_length=window_length, polyorder=polyorder)

    omega_scaled_data = df[['omega_scaled']].to_numpy().flatten()[start_idx:end_idx]
    omega_scaled_filtered = savgol_filter(omega_scaled_data, window_length=window_length, polyorder=polyorder)

    Q_heat_scaled_data = df[['Q_heat_scaled']].to_numpy().flatten()[start_idx:end_idx]
    Q_heat_scaled_filtered = savgol_filter(Q_heat_scaled_data, window_length=window_length, polyorder=polyorder)



    temperatures_tensor = torch.tensor(temperatures_filtered, dtype=torch.float32)
    current_tensor = torch.tensor(current_filtered,  dtype=torch.float32)
    omega_scaled_tensor = torch.tensor(omega_scaled_filtered, dtype=torch.float32)
    Q_heat_scaled_tensor = torch.tensor(Q_heat_scaled_filtered, dtype=torch.float32)

    temperatures_data = temperatures_filtered
    current_data = current_filtered
    omega_scaled_data = omega_scaled_filtered
    Q_heat_scaled_data = Q_heat_scaled_filtered
    dt = 5

    T_bat = T_bat_scale*temperatures_data[0]
    T_preds_nn = []
    residuals = []
    for t in range(n_rows - 1):
        T_env =T_env

        current = current_scale * current_data[t]
        Q_heat = Q_heat_scale * Q_heat_scaled_data[t]
        omega = omega_scale * omega_scaled_data[t]

        mdot_c = density_coolant * pump_displacement * omega
        NTU_bat = (alpha_3 * hA_bat) / (mdot_c * c_coolant + 1e-3)
        T_clin = T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3)
        T_clout = (T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat
        Q_cool = mdot_c * c_coolant * (T_clout - T_clin)
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))


        mlp_input = torch.tensor([[T_bat/T_bat_scale, current/current_scale, omega/omega_scale, Q_heat/Q_heat_scale]],
                                dtype=torch.float32)
        
        
        residual = residual_mlp(mlp_input).detach().item()
        residuals.append(residual)
        T_bat_pred_nn = T_bat + dt*(T_bat_dot_model + alpha*lambda_nn * residual)
        T_preds_nn.append(T_bat_pred_nn)
        T_bat = T_bat_pred_nn 

    T_bat = T_bat_scale*temperatures_data[0]
    T_preds_nom = []

    for t in range(n_rows - 1):
        T_env = T_env


        current = current_scale * current_data[t]
        Q_heat = Q_heat_scale * Q_heat_scaled_data[t]
        omega = omega_scale * omega_scaled_data[t]

        mdot_c = density_coolant * pump_displacement * omega
        NTU_bat = (alpha_3 * hA_bat) / (mdot_c * c_coolant + 1e-3)
        T_clin = T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3)
        T_clout = (T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat
        Q_cool = mdot_c * c_coolant * (T_clout - T_clin)
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))

        T_bat_pred = T_bat + dt*T_bat_dot_model
        T_preds_nom.append(T_bat_pred)
        T_bat = T_bat_pred  

    T_true = T_bat_scale*temperatures_data[:]  
    T_preds_nom = np.insert(T_preds_nom, 0, T_bat_scale*temperatures_data[0])
    T_preds_nn = np.insert(T_preds_nn, 0, T_bat_scale*temperatures_data[0])

    mse_nn = mean_squared_error(T_preds_nn, T_true)
    mse_nom = mean_squared_error(T_preds_nom, T_true)

    print("MSE Neural Network:", mse_nn)
    print("MSE Nominal Model:", mse_nom)
    # Saving pretrained network
    torch.save(residual_mlp.state_dict(), "Pretrained_Networks/heating_pretrain_scaled_network_4_input_new_loss_filtered_tanh_jac_penalty_phys_penalty_FINAL_FINAL.pth")

    plt.figure(1)
    plt.plot(T_true, label='True')
    plt.plot(T_preds_nom, label='Nominal')
    plt.plot(T_preds_nn, label='Neural Network')
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Battery Temp [K]")

    plt.figure(2)
    plt.plot(residuals)
    plt.show()



if __name__ == "__main__":
    main()
    


