import casadi as cs
import numpy as np

T_bat_ss = 20.5 + 273.15
omega_ss = 100 # Power into heater/cooler given as efficiency*Pin
Q_heat = cs.MX.sym('Q_heat') # Pump control (rpm that is converted to kg/s)
current_ss = 12.35
# Parameters
m_battery = 20*2.5*4
c_battery = 795
c_coolant = 3500
density_coolant = 1050
        
pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
R_battery = 4*20*0.0128 # Battery resistance
C_battery = 28*3600 # in coloumb
hA_bat = 2500

T_env = 0+ 273.15

#[0.637968, 1, 0.980682, 0.981036]
alpha_0 = 0.637968 #0.65
alpha_1 = 1 # 0.99 #0.998427 #0.99
alpha_2 = 0.980682  #0.999686 #.97
alpha_3 = 0.981036
gamma = 7.59853
        
mdot_c = density_coolant*pump_displacement*omega_ss
# Dynamics
        
# Get the cooler in and out temps
NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
T_clin= (T_bat_ss + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
T_clout = ((T_clin - T_bat_ss) * alpha_2*np.exp(-NTU_bat) + T_bat_ss)
        
        
Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

# Now for the actual calculations 
T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current_ss**2 * R_battery - Q_cool + gamma*(T_env - T_bat_ss))

T_bat_dot_model = cs.Function("T_bat_model", [Q_heat], [T_bat_dot_model], ["q_heat"], ["ode"])

G = cs.rootfinder('G', 'newton', T_bat_dot_model)
val = G()
print(val['q_heat'].full().item())