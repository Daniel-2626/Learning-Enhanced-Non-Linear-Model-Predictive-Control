import numpy as np
import casadi as cs
from itertools import combinations_with_replacement
from time import time
from functools import partial # What does this do

import pysindy as ps
import pandas as pd
import os
import matplotlib.pyplot as plt
csv_path = os.path.join(os.path.dirname(__file__), 'residuals_nom.csv')
df = pd.read_csv(csv_path)
CELSIUS_TO_KELVIN = 273.15
class SymbolicPolynomialLibrary:
    def __init__(self):
        pass

    def fit(self, variables, max_degree, include_bias =True):
        self.variables = variables
        self.max_degree = max_degree
        self.include_bias = include_bias
        self.feature_expressions = []

        if self.include_bias:
            self.feature_expressions.append(1)

        for deg in range(1, self.max_degree +1):
            for combo in combinations_with_replacement(self.variables, deg):
                feature_expression = 1
                for var in combo:
                    feature_expression *= var
                self.feature_expressions.append(feature_expression)
        return self.feature_expressions

    def get_feature_map(self):
        return self.feature_expressions

    def print_feature_names(self):
        print("symbolic terms of the feature library (degree {})".format(self.max_degree))

        for feature in self.feature_expressions:
            print(feature, end=",")
        print()



T_bat_scale = 100
current_scale = 25
omega_scale = 100
Q_heat_scale = 1000
T_bat_normalized = cs.MX.sym('T_bat_normalized')
T_bat = T_bat_scale * T_bat_normalized

omega_normalized = cs.MX.sym('omega_norm')
omega = omega_scale * omega_normalized #cs.MX.sym('omega') # Power into heater/cooler given as efficiency*Pin
Q_heat_normalized = cs.MX.sym('Q_heat_norm')
Q_heat = Q_heat_scale * Q_heat_normalized #cs.MX.sym('Q_heat') # Pump control (rpm that is converted to kg/s)
current_normalized = cs.MX.sym('current_normalized')
current = current_scale*current_normalized
T_env = 12.5 + CELSIUS_TO_KELVIN
# Parameters
m_battery = 20*2.5*4
c_battery = 795
c_coolant = 3500
density_coolant = 1050
        
pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
R_battery = 4*20*0.0128 # Battery resistance
C_battery = 28*3600 # in coloumb
hA_bat = 2500 # 2500

# Identified
alpha_0 = 0.635039 # 0.637968 
alpha_1 = 0.915692 # 1 
alpha_2 = 0.919681 # 0.980682  
alpha_3 = 1.47275 # 0.981036
gamma = 7.38325 # 7.
        
mdot_c = density_coolant*pump_displacement*omega
# Dynamics
        
# Get the cooler in and out temps
NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
T_clin= (T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
T_clout = ((T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat)
        

Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

# Now for the actual calculations 
T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))/T_bat_scale
       
X_dot_nominal = cs.vertcat(T_bat_dot_model)
T_bat_dot_function = cs.Function("f_model", [T_bat_normalized, current_normalized, omega_normalized, Q_heat_normalized], 
[T_bat_dot_model], ["T_bat_normalized", "current_normalized", "omega_normalized", "Q_heat_normalized"], ["ode"])

f_expl = X_dot_nominal 
start_idx = 0
end_idx = 200
rows = end_idx - start_idx
temperatures_data = df[['T_bat']].to_numpy().flatten()[start_idx:end_idx]

#temperatures_filtered = savgol_filter(temperatures_data, window_length=window_length, polyorder=polyorder)

current_data = df[['current']].to_numpy().flatten()[start_idx:end_idx]
#current_filtered = savgol_filter(current_data, window_length=window_length, polyorder=polyorder)

omega_scaled_data = df[['omega_scaled']].to_numpy().flatten()[start_idx:end_idx]
#omega_scaled_filtered = savgol_filter(omega_scaled_data, window_length=window_length, polyorder=polyorder)

Q_heat_scaled_data = df[['Q_heat_scaled']].to_numpy().flatten()[start_idx:end_idx]

dt = 5
predictions = []
T_bat = temperatures_data[0]
for k in range(rows):

    current = current_data[k]
    omega = omega_scaled_data[k]
    Q_heat =Q_heat_scaled_data[k]

    T_bat_pred = T_bat + dt*T_bat_dot_function(T_bat, current, omega, Q_heat)
    T_bat = T_bat_pred
    predictions.append(T_bat_pred)

predictions = np.array(predictions)
predictions = predictions.flatten()
plt.plot(predictions)
true_temperatures = temperatures_data[1:]
plt.plot(true_temperatures)
plt.show()


# state symbolic variables 
states = cs.vertcat(T_bat_normalized)


controls = cs.vertcat(current_normalized, omega_normalized, Q_heat_normalized)

# Feature map
max_degree = 3

include_bias = False
states_concat = [T_bat_normalized,current_normalized,omega_normalized, Q_heat_normalized]
library = SymbolicPolynomialLibrary()
library.fit(variables=states_concat, max_degree=max_degree, include_bias=include_bias)
states_mapped = library.get_feature_map()
library.print_feature_names()
feature_names = library.get_feature_names()

x_train = temperatures_data.reshape((-1,1))
u_train = np.hstack([current_data, omega_scaled_data, Q_heat_scaled_data])
u_train = np.transpose(u_train)
ssr_optimizer = ps.SSR(alpha=0.05,criteria="model_residual",kappa=1e-1) # Optimiser SSR Greedy Optimiser
library = ps.PolynomialLibrary(degree=2,include_bias=False).fit(x_train) # Feature Library
model = ps.SINDy(optimizer=ssr_optimizer,feature_library=library,
                            feature_names=feature_names,discrete_time=False)
model.fit(x=x_train,u=u_train,t=dt)


print("All terms in the feature library")
print(self.feature_names)
model_name = "BTM"
print("Model Discovery (%s) :" % model_name)
model.print()