import scipy.linalg
import numpy as np
import casadi as cs
        
T_bat = cs.MX.sym('T_bat')

omega_norm = cs.MX.sym('omega_norm')
omega = 100 * omega_norm
Q_heat_norm = cs.MX.sym('Q_heat_norm')
Q_heat = 1000 * Q_heat_norm #cs.MX.sym('Q_heat')

U = cs.vertcat(omega_norm, Q_heat_norm)
        
# Parameters
m_battery = 20*2.5*4
c_battery = 795
c_coolant = 3500
density_coolant = 1050
        
pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
R_battery = 4*20*0.0128 # Battery resistance
C_battery = 28*3600 # in coloumb
hA_bat = 2500        

alpha_0 = 0.635039 # 0.637968 
alpha_1 = 0.915692 # 1 
alpha_2 = 0.919681 # 0.980682  
alpha_3 = 1.47275 # 0.981036
gamma = 7.38325 # 7.59853
        
mdot_c = density_coolant*pump_displacement*omega
# Dynamics
        
# Get the cooler in and out temps
NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
T_clin= (T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
T_clout = ((T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat)
Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

T_env = -10 + 273.15

# Now for the actual calculations 
T_bat_dot_model = alpha_0/(m_battery*c_battery) * (- Q_cool + gamma*(T_env - T_bat))
f_T_bat_dot = cs.Function("f_model", [T_bat,U], [T_bat_dot_model], ["x", "u"], ["ode"])

T_bat_target = 20.5 + 273.15
omega_ss = 1.00
Q_heat_ss = 2.30

jac_T_bat = cs.jacobian(T_bat_dot_model, T_bat)
jac_T_bat_fun = cs.Function("f_dT_bat", [T_bat,U], [jac_T_bat], ["x", "u"], ["f_dT_bat"])

jac_U = cs.jacobian(T_bat_dot_model, U)
jac_U_fun = cs.Function("f_dU", [T_bat,U], [jac_U], ["x", "u"], ["f_dU"])

A = jac_T_bat_fun(T_bat_target+273.15, [omega_ss, Q_heat_ss]).full()
print(A)
B = jac_U_fun(T_bat_target+273.15, [omega_ss, Q_heat_ss]).full()
B = B.reshape((1,2))
print(B)

a = A
b = B
q = 10000
r = np.diag([1,1])
cost_to_go = scipy.linalg.solve_continuous_are(a = a, b = b, q = q, r = r).item()
print(cost_to_go)