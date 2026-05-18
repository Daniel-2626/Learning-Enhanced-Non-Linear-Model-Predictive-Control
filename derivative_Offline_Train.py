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
csv_path = os.path.join(os.path.dirname(__file__), 'residuals_matched_heating.csv')
df = pd.read_csv(csv_path)
print(df.head())
#training_data = df[['T_bat', 'current', 'omega_scaled', 'Q_heat_scaled']].to_numpy()
#training_data = df[['T_bat', 'current', 'Q_heat_scaled']].to_numpy()[:2000]

#training_data[:,1] = 25*training_data[:,1]e
#training_data[:,0] = end*training_data[:,0]
#training_data = df[['current']].to_numpy()[end:400]

#training_data[:,0] = end*training_data[:,0]
#training_data[:,1] = end*training_data[:,1]
#training_data = df[['T_bat', 'current', 'Q_heat_scaled']].to_numpy()

#training_data = df[['h1', 'h2', 'u']].to_numpy()
#print(training_data)

#residuals = df[['residual_1', 'residual_2']].to_numpy()

# input and disturbance values (not symbolics because know what happened)
# Not sure if should take current values or last, but figure that at this moment the change is happening because of the last values
dt = 5

start = 0
end = -1
polyorder = 2
window_length = 10

omega_scale = 100
Q_heat_scale = 1000
T_bat_scale = 100
current_scale = 25

input_data = df[['T_bat', 'current', 'omega_scaled', 'Q_heat_scaled']].to_numpy()[start:end]
temperatures_data = df[['T_bat']].to_numpy().flatten()[start:end]

temperatures_filtered = savgol_filter(temperatures_data, window_length=window_length, polyorder=polyorder)

current_data = df[['current']].to_numpy().flatten()[start:end]
current_filtered = savgol_filter(current_data, window_length=window_length, polyorder=polyorder)

omega_scaled_data = df[['omega_scaled']].to_numpy().flatten()[start:end]
omega_scaled_filtered = savgol_filter(omega_scaled_data, window_length=window_length, polyorder=polyorder)

Q_heat_scaled_data = df[['Q_heat_scaled']].to_numpy().flatten()[start:end]
Q_heat_scaled_filtered = savgol_filter(Q_heat_scaled_data, window_length=window_length, polyorder=polyorder)

training_data = np.vstack([temperatures_filtered,current_filtered, omega_scaled_filtered,Q_heat_scaled_data])
training_data = np.transpose(training_data)
n_rows, n_columns = input_data.shape
CELSIUS_TO_KELVIN = 273.15

dT_bat_model_filtered_before = []
for row in range(0,n_rows):
    T_env = 12.5 + CELSIUS_TO_KELVIN
    omega = omega_scale*omega_scaled_filtered[row]
    Q_heat = Q_heat_scale*Q_heat_scaled_filtered[row]
    #current = self.current_last
    #T_bat = self.T_bat_last
                
                
    current = current_scale*current_filtered[row]
    T_bat = T_bat_scale*temperatures_filtered[row]


    # Caluclate derivative with model
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
            
    mdot_c = density_coolant*pump_displacement*omega
    # Dynamics
            
    # Get the cooler in and out temps
    NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
    T_clin= (T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
    T_clout = ((T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat)
          
    Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

    T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))
    dT_bat_model_filtered_before.append(T_bat_dot_model)



residuals = df[['residual']].to_numpy().flatten()[start:end]



dT_bat_savgol = 100*savgol_filter(temperatures_data, window_length=window_length, polyorder=polyorder, deriv = 1, delta = dt)
dT_bat_model = df[['T_bat_dot_model']].to_numpy().flatten()[start:end]
print(dT_bat_model)

dT_bat_model_filtered_before = np.array(dT_bat_model_filtered_before)
residuals_filtered_before = dT_bat_savgol - dT_bat_model_filtered_before


plt.figure(1)
plt.plot(dT_bat_model)
plt.plot(dT_bat_model_filtered_before)
plt.legend(("dT bat model", "dT_bat_model filtered"))

plt.figure(2)
dT_bat_euler = df[['T_bat_dot_euler']].to_numpy().flatten()[start:end]
plt.plot(dT_bat_euler)
plt.plot(dT_bat_savgol)
plt.legend(("dT bat euler raw", "dT_bat_savgol filtered"))

plt.figure(3)
plt.plot(residuals)
plt.plot(residuals_filtered_before)
plt.legend(("residuals raw", "residuals savgol filtered"))

plt.show()


print("average", np.mean(residuals))

residual_scale = 1
residuals = residual_scale * residuals
residuals = residuals.reshape((-1,1))
residuals_filtered_before = residual_scale*residuals_filtered_before.reshape((-1,1))
print("training data shape", training_data.shape)
print("residuals shape", residuals_filtered_before.shape)
N_data_points = len(residuals_filtered_before)
train_size = 0.5
test_size = 1 - train_size
X_train = training_data[:int(N_data_points*train_size),:]
X_test = training_data[int(N_data_points*train_size):]
y_train = residuals_filtered_before[:int(N_data_points*train_size),:]
y_test = residuals_filtered_before[int(N_data_points*train_size):]

#X_train, X_test, y_train, y_test = train_test_split(training_data, residuals_filtered_before, test_size=0.3)

class MLP(nn.Module):
    def __init__(self, input_dim=4, output_dim=2, hidden_dim=16, num_layers=2):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor([0.001]), requires_grad=False)        

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        net = self.net(x)
        #return net 
        return self.tau * torch.tanh(net)

# MAML Training Loop

def main():
    device = torch.device("cpu")
    
    # Hyperparameters
    learning_rate = 1e-2
    input_dim = 4
    output_dim = 1
    hidden_dim = 16
    num_layers = 2

    model = MLP(input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    residual_mlp = MLP(input_dim = input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers) # the network
    start = time.time()
    for param in residual_mlp.parameters():
        param.requires_grad = False
    residual_mlp = residual_mlp
    residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=learning_rate, weight_decay=0.1) # lr = learning rate, the optimizer
    #residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=learning_rate) # lr = learning rate, the optimizer
    
    residual_criterion = nn.MSELoss()
    print(y_train.shape)
    X_batch = torch.tensor(X_train, dtype=torch.float32)
    y_target = torch.tensor(y_train.copy(), dtype=torch.float32)
    confidence = np.sqrt(np.power(y_train,2).mean())
    print('confidence', confidence)
    #np.sqrt(np.power(dynamics_savgol-dynamics_model,2).mean())
    with torch.no_grad():
        residual_mlp.tau.copy_(torch.tensor([confidence]))            
        print("confidence", confidence)
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
        nn.utils.clip_grad_norm_(
            residual_mlp.parameters(),
            max_norm= 0.01
        )
        residual_optimizer.step() # one optimization step to update parameters
    end = time.time()
    print("elapsed", 1000*(start-end))
    num_params = 0

    for p in residual_mlp.parameters(): 
        p.requires_grad = False
        num_params += len(p.flatten())
        #print(p)
    print("num_params", num_params)
    torch.save(residual_mlp.state_dict(), "heating_pretrain_deriv_network_4_input_tanh_FINAL.pth")

    test_data = torch.tensor(X_test, dtype=torch.float32)
    residual_mlp.eval()
    with torch.no_grad():
        outputs = residual_mlp(test_data)
        target_predicted = np.array(outputs.squeeze().tolist())
    mse = mean_squared_error(target_predicted, y_test)
    print(y_test.size)
    len_test = len(y_test)
    null_prediction = np.zeros((len_test,))
    print("MSE", mse)
    mse_null_hypothesis = mean_squared_error(y_test, null_prediction)


    print("MSE null hypothesis", mse_null_hypothesis)
    print("MSE/MSE null hypothesis", mse/mse_null_hypothesis)
    plt.figure(3)
    plt.plot(target_predicted)
    plt.plot(y_test)

    plt.figure(4)
    print(residuals.shape)
    print(target_predicted.shape)
    plt.plot(y_test.flatten())
    plt.plot(y_test.flatten()-target_predicted.flatten())
    
    plt.legend(("y_test", "y_test-y_nn"))

    with torch.no_grad():
        input_data_tensor = torch.tensor(training_data, dtype=torch.float32)
        outputs = residual_mlp(input_data_tensor)
        target_predicted = np.array(outputs.squeeze().tolist())
    plt.figure(5)
    plt.plot(target_predicted)
    plt.plot(residuals_filtered_before)
    plt.legend(("target predicted", "residuals filtered before"))

    #plt.ylim(0, 2/end) # Set y-axis  


    plt.show()
    #for var_name in residual_optimizer.state_dict():
    #    print(var_name, '\t', residual_optimizer.state_dict()[var_name])

 
if __name__ == "__main__":
    main()
    

