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
csv_path = os.path.join(os.path.dirname(__file__), 'matched_residuals_thermal_management.csv')
df = pd.read_csv(csv_path)
print(df.head())

class MLP(nn.Module):

    def __init__(self, input_dim=4, output_dim=2, hidden_dim=16, num_layers=2):
        super().__init__()
  
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        self.tau = nn.Parameter(torch.tensor([0.001]), requires_grad=False)        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        with torch.no_grad():
            self.net[-1].weight.fill_(0.0)
            self.net[-1].bias.fill_(0.0)
    def forward(self, x):

        net = self.net(x)
        #return net
        return self.tau * torch.tanh(net) # Kinda BS cuz then our residual NN is basically just a tanh function
        # Or in other words: An expected value in two parts
        # But not really. Not necssarily tanh in time, just in inputs.

# MAML Training Loop

def plot_control_jacobians(model, T_bat_fixed, current_fixed, omega_scale, Q_heat_scale):
    model.eval() # Set to evaluation mode
    
    # Create a range for Q_heat (normalized -1 to 1)
    q_range = torch.linspace(-1, 1, 100).reshape(-1, 1)
    # Fixed inputs (normalized)
    t_in = torch.full_like(q_range, T_bat_fixed / 100.0)
    i_in = torch.full_like(q_range, current_fixed / 25.0)
    w_in = torch.full_like(q_range, 0.5) # Fixed pump at 50%
    
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
    #input_data = df[['T_bat', 'current', 'omega_scaled', 'Q_heat_scaled']].to_numpy()[start_idx:end_idx]
    temperatures_data = df[['T_bat']].to_numpy().flatten()[start_idx:end_idx]

    temperatures_filtered = savgol_filter(temperatures_data, window_length=window_length, polyorder=polyorder)

    current_data = df[['current']].to_numpy().flatten()[start_idx:end_idx]
    current_filtered = savgol_filter(current_data, window_length=window_length, polyorder=polyorder)

    omega_scaled_data = df[['omega_scaled']].to_numpy().flatten()[start_idx:end_idx]
    omega_scaled_filtered = savgol_filter(omega_scaled_data, window_length=window_length, polyorder=polyorder)

    Q_heat_scaled_data = df[['Q_heat_scaled']].to_numpy().flatten()[start_idx:end_idx]
    Q_heat_scaled_filtered = savgol_filter(Q_heat_scaled_data, window_length=window_length, polyorder=polyorder)


    # Hyperparameters
    learning_rate = 1e-3
    input_dim = 4
    output_dim = 1
    hidden_dim = 32
    num_layers = 2

    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    residual_mlp = MLP(input_dim = input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers) # the network
    start = time.time()

    for param in residual_mlp.parameters():
        param.requires_grad = False
    residual_mlp = residual_mlp
    residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=learning_rate, weight_decay=0.01) # lr = learning rate, the optimizer    

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
    alpha_0 = 0.635039 #0.65
    alpha_1 = 0.915692 # 0.99 #0.998427 #0.99
    alpha_2 = 0.919681  #0.999686 #.97
    alpha_3 = 1.47275
    gamma = 7.38325
    dt = 5
    T_env = (temperatures_data[0]*T_bat_scale) 
    print("T_env", T_env)

    n_rows = end_idx - start_idx
    dynamics_model = []
    for k in range(n_rows):
        omega_scaled = omega_scaled_data[k]
        omega = omega_scale*omega_scaled
        Q_heat_scaled = Q_heat_scaled_data[k]
        Q_heat = Q_heat_scale*Q_heat_scaled
                            
        current_scaled = current_filtered[k]            
        current = current_scale*current_scaled
                
        T_bat = T_bat_scale* temperatures_data[k]

        mdot_c = density_coolant*pump_displacement*omega
        # Dynamics               
        # Get the cooler in and out temps
        NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
        T_clin= (T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
        T_clout = ((T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat)
                 
        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))
        dynamics_model.append(T_bat_dot_model)
    # The savgol derivative is assumed to be true dynamics
    dynamics_savgol = T_bat_scale*savgol_filter(temperatures_data, window_length=window_length, polyorder=polyorder, deriv = 1, delta = dt)

    #confidence = np.sqrt(np.power(dynamics_savgol-dynamics_model,2).mean())
    confidence = 0.001
    #confidence = np.clip(confidence, 0.001, 0.05)
    with torch.no_grad():
        residual_mlp.tau.copy_(torch.tensor([confidence]))
    print("confidence", confidence)
    for p in residual_mlp.parameters(): p.requires_grad = True

    #temperatures_tensor = torch.tensor(temperatures_data, dtype=torch.float32)
    #current_tensor = torch.tensor(current_data,  dtype=torch.float32)
    #omega_scaled_tensor = torch.tensor(omega_scaled_data, dtype=torch.float32)
    #Q_heat_scaled_tensor = torch.tensor(Q_heat_scaled_data, dtype=torch.float32)

    temperatures_tensor = torch.tensor(temperatures_data, dtype=torch.float32)
    current_tensor = torch.tensor(current_filtered,  dtype=torch.float32)
    omega_scaled_tensor = torch.tensor(omega_scaled_data, dtype=torch.float32)
    Q_heat_scaled_tensor = torch.tensor(Q_heat_scaled_data, dtype=torch.float32)
    regularization = 0
    #lambda_nn =0.1
    T_rollout = 20
    lambda_nn = 1
    lambda_phys = 1
    lambda_jac = 0.1
    alpha = 1
    epsilon = 0.2
    noise_std = 0.001
    # TBPT
 
    for epoch in range(200):
        residual_optimizer.zero_grad() # optimizer object
        loss = 0
        residual_loss = 0
        T_bat = T_bat_scale*temperatures_tensor[0]
        weight = 1.02
        # We'll rollout for 10 steps and then update T
        # There is no teacher forcing here I believe
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
                mdot_c = density_coolant*pump_displacement*omega
                # Dynamics
                T_bat_noisy = T_bat + torch.randn_like(T_bat) * noise_std 
               
                
                # Get the cooler in and out temps
                NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
                T_clin= (T_bat_noisy + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
                T_clout = ((T_clin - T_bat_noisy) * alpha_2*np.exp(-NTU_bat) + T_bat_noisy)
                    
                Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

                T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat_noisy))
                
                #mlp_input = torch.tensor([[T_bat/T_bat_scale, current/current_scale, Q_heat/Q_heat_scale]], dtype=torch.float32)                
                #mlp_input = torch.tensor([[T_bat/T_bat_scale, omega/omega_scale, Q_heat/Q_heat_scale]], dtype=torch.float32)                
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

                jacobian_T = jacobian[:, 0]
                #print(jacobian)
                # jacobian[0]

                jacobian_penalty = jacobian_T.pow(2).mean()
                phys_penalty = 0 #(F.relu((-Q_heat/Q_heat_scale).reshape(1,1)*residual))**2 #+ F.relu((current/current_scale).reshape(1,1)*residual)
                # sigma_max = torch.linalg.norm(jacobian, ord=2)
                #jacobian_penalty = sigma_max**2 Spectral norm instead of frobenius
                # also consider just doing on dJ/dx

                T_bat_pred = T_bat_noisy + dt*(T_bat_dot_model+ lambda_nn*residual)
                error = T_bat_next - T_bat_pred 
                loss +=  weight**i * torch.where(torch.abs(error)< epsilon, torch.zeros_like(error), error**2) 
                loss += lambda_jac*jacobian_penalty
                #print("phys loss", lambda_phys * phys_penalty)
                #print("traj loss one step", weight**i * (T_bat_next - T_bat_pred)**2 )
                loss += lambda_phys * phys_penalty
                residual_loss = residual_loss # + residual.pow(2).mean()
                T_bat = T_bat_pred.squeeze()
            T_bat = T_bat.detach().clone()
        loss += regularization * residual_loss/n_rows 
        loss.backward() # calculates gradient
        nn.utils.clip_grad_norm_(
            residual_mlp.parameters(),
            max_norm= 0.1
        ) # Try with a lower value

        #with torch.no_grad():
        #    for p in residual_mlp.parameters():
        #        p.clamp_(-0.01,0.01)

    
        residual_optimizer.step() # one optimization step to update parameters

        if epoch % 20 == 0:
            print("Epoch {}: Training loss: {}".format(epoch, loss.item()))
    end = time.time()
    print("elapsed", 1000*(start-end))
    num_params = 0

    #for p in residual_mlp.parameters(): 
    #    p.requires_grad = False
    #    num_params += len(p.flatten())
    #    print(p)
    print("num_params", num_params)
    #temperatures_data = df[['T_bat']].to_numpy().flatten()[start_idx:end_idx]
    #current_data = df[['current']].to_numpy().flatten()[start_idx:end_idx]
    #omega_scaled_data = df[['omega_scaled']].to_numpy().flatten()[start_idx:end_idx]
    #Q_heat_scaled_data = df[['Q_heat_scaled']].to_numpy().flatten()[start_idx:end_idx]

    polyorder = 2
    window_length = 10
    start_idx = 200
    end_idx = 400
    n_rows = end_idx - start_idx
    #input_data = df[['T_bat', 'current', 'omega_scaled', 'Q_heat_scaled']].to_numpy()[start_idx:end_idx]
    temperatures_data = df[['T_bat']].to_numpy().flatten()[start_idx:end_idx]

    temperatures_filtered = savgol_filter(temperatures_data, window_length=window_length, polyorder=polyorder)

    current_data = df[['current']].to_numpy().flatten()[start_idx:end_idx]
    current_filtered = savgol_filter(current_data, window_length=window_length, polyorder=polyorder)

    omega_scaled_data = df[['omega_scaled']].to_numpy().flatten()[start_idx:end_idx]
    omega_scaled_filtered = savgol_filter(omega_scaled_data, window_length=window_length, polyorder=polyorder)

    Q_heat_scaled_data = df[['Q_heat_scaled']].to_numpy().flatten()[start_idx:end_idx]
    Q_heat_scaled_filtered = savgol_filter(Q_heat_scaled_data, window_length=window_length, polyorder=polyorder)


    temperatures_tensor = torch.tensor(temperatures_data, dtype=torch.float32)
    current_tensor = torch.tensor(current_filtered,  dtype=torch.float32)
    omega_scaled_tensor = torch.tensor(omega_scaled_data, dtype=torch.float32)
    Q_heat_scaled_tensor = torch.tensor(Q_heat_scaled_data, dtype=torch.float32)

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

        #mlp_input = torch.tensor([[T_bat / T_bat_scale, current / current_scale, Q_heat / Q_heat_scale]],
        #                        dtype=torch.float32)
        #mlp_input = torch.tensor([[T_bat/T_bat_scale, omega/omega_scale, Q_heat/Q_heat_scale]],
        #                       dtype=torch.float32)
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

    T_true = T_bat_scale*temperatures_data[1:]  
    mse_nn = mean_squared_error(T_preds_nn, T_true)
    mse_nom = mean_squared_error(T_preds_nom, T_true)

    print("MSE Neural Network:", mse_nn)
    print("MSE Nominal Model:", mse_nom)
    torch.save(residual_mlp.state_dict(), "economic_pretrain_FINAL.pth")

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

    plot_control_jacobians(residual_mlp, 293.15, 40, 100, 1000)

if __name__ == "__main__":
    main()
    


