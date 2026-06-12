import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from collections import defaultdict
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


class MLP(nn.Module):
    def __init__(self, input_dim=4, output_dim=1, hidden_dim=16, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        self.tau = nn.Parameter(torch.tensor([0.001]), requires_grad=False)        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        net = self.net(x)
        return self.tau * torch.tanh(net)

def plot_control_jacobians(model, T_bat_fixed, current_fixed, omega_scale, Q_heat_scale):
    model.eval() 
    
    q_range = torch.linspace(-1, 1, 100).reshape(-1, 1)
    t_in = torch.full_like(q_range, T_bat_fixed / 100.0)
    i_in = torch.full_like(q_range, current_fixed / 25.0)
    w_in = torch.full_like(q_range, 0.5) 
    
    inputs = torch.cat([t_in, i_in, w_in, q_range], dim=1)
    inputs.requires_grad_(True)
    
    residual = model(inputs)
    
    grads = torch.autograd.grad(
        outputs=residual,
        inputs=inputs,
        grad_outputs=torch.ones_like(residual),
        create_graph=False
    )[0]
    
    d_res_dq = grads[:, 3].detach().numpy()
    d_res_dw = grads[:, 2].detach().numpy()
    res_val = residual.detach().numpy()

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
    # Load dataset of residuals
    csv_path = os.path.join(os.path.dirname(__file__), 'residuals_matched_heating_T_env_5.csv')
    df = pd.read_csv(csv_path)
    print(df.head())
    device = torch.device("cpu")
    start_idx = 0
    # Because of how batching is done, and since we need one extra temperature for the last prediction, it is most efficent
    # data wise to collect + 1 data points
    # E.g. (start_idx = 0, end_idx = 200) --> 200 datapoints 
    # T_rolling = 20, need batches of 20 (T_rolling) 
    # When predicting, for the last batch, need 1 extra datapoint
    # Having 200 data points means we can only use 9 batches * 20 = 180 datapoints
    # Having 201 data points means we can use 10 batches * 200 = 200 datapoints
    end_idx = 601
    n_rows = end_idx - start_idx

    CELSIUS_TO_KELVIN = 273.15

    omega_scale = 4000*2*np.pi/60
    Q_heat_scale = 4000
    T_bat_scale = CELSIUS_TO_KELVIN + 50
    current_scale = 100

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

    residual_mlp = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)

    start = time.time()

    #for param in residual_mlp.parameters():
    #    param.requires_grad = False
    residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=learning_rate, weight_decay=0.1)

    # Parameters
    m_battery = 20*2.5*4
    c_battery = 795
    c_coolant = 3500
    density_coolant = 1050
                    
    pump_displacement = 1/(2*np.pi)*40/(100**3) 
    R_battery = 4*20*0.0128 
    hA_bat = 2500
    alpha_0 = 0.635039 
    alpha_1 = 0.915692 
    alpha_2 = 0.919681  
    alpha_3 = 1.47275
    gamma = 7.38325
    dt = 5
    T_env = (temperatures_data[0]*T_bat_scale)
    print("T_env", T_env)
    confidence = 0.001
    with torch.no_grad():
        residual_mlp.tau.copy_(torch.tensor([confidence]))            
    for p in residual_mlp.parameters(): p.requires_grad = True

    # Convert to tensors
    temperatures_tensor = torch.tensor(temperatures_filtered, dtype=torch.float32)
    current_tensor = torch.tensor(current_filtered,  dtype=torch.float32)
    omega_scaled_tensor = torch.tensor(omega_scaled_filtered, dtype=torch.float32)
    Q_heat_scaled_tensor = torch.tensor(Q_heat_scaled_filtered, dtype=torch.float32)

    # Vectorization

    T_rollout = 20
    # When doing simulation, will do rollout over 20 steps
    # Need to split data into chunks of 20 steps
    max_len = ((n_rows - 1) // T_rollout) * T_rollout
    batch_size = max_len // T_rollout
    
    print("max_len", max_len)
    print("batch_size", batch_size)

    # Reshape into tables of batch_size, T_rollout .view similar to .reshape in numpy
    temp_table = temperatures_tensor[:max_len].view(batch_size, T_rollout)
    temp_next_table = temperatures_tensor[1:max_len+1].view(batch_size, T_rollout)
    current_table = current_tensor[:max_len].view(batch_size, T_rollout)
    omega_table = omega_scaled_tensor[:max_len].view(batch_size, T_rollout)
    Q_heat_table = Q_heat_scaled_tensor[:max_len].view(batch_size, T_rollout)

    regularization = 0
    lambda_nn = 1
    lambda_phys = 1
    lambda_jac = 0.1
    alpha = 1
    epsilon = 0.1
    noise_std = 0.005
    weight = 1.02

    for epoch in range(1000):
        residual_optimizer.zero_grad()
        loss = 0
        residual_loss = 0

        # True T_bat for all batches
        T_bat = T_bat_scale * temp_table[:, 0]

        # Rollout over all trajectory chunks at once
        for i in range(T_rollout):
            # Extract current time step across the whole batch
            omega_scaled = omega_table[:, i]
            current_scaled = current_table[:, i]
            Q_heat_scaled = Q_heat_table[:, i]

            # Target for loss function
            T_bat_next = T_bat_scale * temp_next_table[:, i]
            #print(T_bat_next.size())

            omega = omega_scale * omega_scaled
            Q_heat = Q_heat_scale * Q_heat_scaled
            current = current_scale * current_scaled

            mdot_c = density_coolant * pump_displacement * omega

            # Dynamics 
            T_bat_noisy = T_bat + torch.randn_like(T_bat) * noise_std

            NTU_bat = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
            T_clin= (T_bat_noisy + alpha_1*(1/(1-torch.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
            T_clout = ((T_clin - T_bat_noisy) * alpha_2*torch.exp(-NTU_bat) + T_bat_noisy)
            Q_cool = mdot_c*c_coolant*(T_clout - T_clin)
            T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat_noisy))
            # Stack inputs 
            mlp_input = torch.stack([
                T_bat_noisy / T_bat_scale,
                current / current_scale,
                omega / omega_scale,
                Q_heat / Q_heat_scale
            ], dim=1)

            mlp_input.requires_grad_(True)
            residual = residual_mlp(mlp_input) # Dim: Batch size x 1

            jacobian = torch.autograd.grad(
                outputs=residual,
                inputs=mlp_input,
                grad_outputs=torch.ones_like(residual),
                create_graph=True,
                retain_graph=True
            )[0]

            jacobian_T = jacobian[:, 0]
            jacobian_penalty = jacobian_T.pow(2).mean()
            #phys_penalty = 0

            # Flatten the residual to have same shape as T_bat_next
            T_bat_pred = T_bat_noisy + dt*(T_bat_dot_model + lambda_nn * residual.squeeze(1))
            error = T_bat_next - T_bat_pred
            step_loss = weight**i * torch.where(torch.abs(error) < epsilon, torch.zeros_like(error), error**2)
            loss += step_loss.mean() 
            loss += lambda_jac * jacobian_penalty
            #loss += lambda_phys * phys_penalty
            
            T_bat = T_bat_pred

        loss += regularization * residual_loss / n_rows 
        loss.backward() 
        nn.utils.clip_grad_norm_(residual_mlp.parameters(), max_norm=0.1)
        residual_optimizer.step() 

        if epoch % 20 == 0:
            print("Epoch {}: Training loss: {}".format(epoch, loss.item()))

    end = time.time()
   
    print("elapsed", 1000*(end-start))
    
    # Evaluation
    
    temperatures_data = temperatures_filtered
    current_data = current_filtered
    omega_scaled_data = omega_scaled_filtered
    Q_heat_scaled_data = Q_heat_scaled_filtered

    T_bat = T_bat_scale * temperatures_data[0]
    T_preds_nn = []
    residuals = []
    
    for t in range(n_rows - 1):
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

    T_bat = T_bat_scale * temperatures_data[0]
    T_preds_nom = []

    for t in range(n_rows - 1):
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
    torch.save(residual_mlp.state_dict(), "heating_pretrain_scaled_network_4_input_new_loss_filtered_tanh_jac_penalty_phys_penalty_FINAL_FINAL_FINAL.pth")

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

    data = {"True": T_true, "Nominal": T_preds_nom}
    #df_simul = pd.DataFrame(data)
    #df_simul.to_csv("nominal_sim.csv", index=False)


if __name__ == "__main__":
    main()