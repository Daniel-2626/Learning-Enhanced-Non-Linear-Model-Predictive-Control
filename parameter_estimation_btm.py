import numpy as np
import casadi as cs
import matplotlib.pyplot as plt


opti = cs.Opti()

N = 2440

alpha_0 = opti.variable()
alpha_1 = opti.variable()
alpha_2 = opti.variable()
T_bat = opti.parameter(N)
omega = opti.parameter(N) # Power into heater/cooler given as efficiency*Pin
Q_heat = opti.parameter(N) # Pump control (rpm that is converted to kg/s)
current = opti.parameter(N)

# Parameters
m_battery = 20*2.5*4
c_battery = 795
c_coolant = 3500
density_coolant = 1050
pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
R_battery = 4*20*0.0128 # Battery resistance
C_battery = 28*3600 # in coloumb
hA_bat = 2500
alpha_0 = 0.65
alpha_1 = 0.99
alpha_2 = .97

mdot_c = density_coolant*pump_displacement*omega
        
# Dynamics
        
# Get the cooler in and out temps
NTU_bat  = (hA_bat) / (mdot_c*c_coolant + 1e-3)
T_clin= alpha_1*(T_bat + 1/(1-np.exp(-NTU_bat))*Q_heat/(mdot_c*c_coolant + 1e-3))
T_clout = alpha_2*((T_clin - T_bat) * np.exp(-NTU_bat) + T_bat)
        
Q_cool = mdot_c*c_coolant*(T_clout - T_clin)


T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool)
f = cs.Function("f", [alpha_0, alpha_1, alpha_2,T_bat, omega, Q_heat, current], [T_bat_dot_model], ["alpha_0","alpha_1", "alpha_2", "T_bat", "omega", "Q_heat", "current"], ["ode"])
        
def RK4(alpha_0, alpha_1, alpha_2, T_bat, omega, Q_heat, current, dt, f):
    K1 = f(alpha_0, alpha_1, alpha_2, T_bat, omega, Q_heat, current)
    K2 = f(alpha_0, alpha_1, alpha_2, T_bat + dt/2 * K1, omega, Q_heat, current)
    K3 = f(alpha_0, alpha_1, alpha_2, T_bat + dt/2 * K2, omega, Q_heat, current)
    K4 = f(alpha_0, alpha_1, alpha_2, T_bat + dt * K3, omega, Q_heat, current)
    next_state = state + (dt/6) * (K1 + 2*K2 + 2*K3 +K4)
    return next_state