import casadi as cs
import numpy as np

# T_bat_ss = 20.5 + 273.15
# omega_ss = 100 # Power into heater/cooler given as efficiency*Pin
# Q_heat = cs.MX.sym('Q_heat') # Pump control (rpm that is converted to kg/s)
# current_ss = 12.35
# # Parameters
# m_battery = 20*2.5*4
# c_battery = 795
# c_coolant = 3500
# density_coolant = 1050
        
# pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
# R_battery = 4*20*0.0128 # Battery resistance
# C_battery = 28*3600 # in coloumb
# hA_bat = 2500

# T_env = 0+ 273.15

# #[0.637968, 1, 0.980682, 0.981036]
# alpha_0 = 0.637968 #0.65
# alpha_1 = 1 # 0.99 #0.998427 #0.99
# alpha_2 = 0.980682  #0.999686 #.97
# alpha_3 = 0.981036
# gamma = 7.59853
        
# mdot_c = density_coolant*pump_displacement*omega_ss
# # Dynamics
        
# # Get the cooler in and out temps
# NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
# T_clin= (T_bat_ss + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
# T_clout = ((T_clin - T_bat_ss) * alpha_2*np.exp(-NTU_bat) + T_bat_ss)
        
        
# Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

# # Now for the actual calculations 
# T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current_ss**2 * R_battery - Q_cool + gamma*(T_env - T_bat_ss))

# T_bat_dot_model = cs.Function("T_bat_model", [Q_heat], [T_bat_dot_model], ["q_heat"], ["ode"])

# G = cs.rootfinder('G', 'newton', T_bat_dot_model)
# val = G()
# print(val['q_heat'].full().item())

## CONFIG
# MPC setup  

CELSIUS_TO_KELVIN = 273.15 
R = np.diag([1, 1])   

N = 1 # number of look ahead steps

t_horizon = 5
step_horizon = t_horizon/N # time between steps in seconds

# Constraints
T_bat_max = 50 + CELSIUS_TO_KELVIN
T_bat_min = -20 + CELSIUS_TO_KELVIN

# State symbolic variables
T_bat = cs.SX.sym('T_bat')
        
states = cs.vertcat(
    T_bat,
)
n_states = states.numel() # Returns amount of elements

# control symbolic variables
omega_norm = cs.SX.sym('omega_norm')
omega_scale = 100
Q_heat_norm = cs.SX.sym('Q_heat_norm')
Q_heat_scale = 1000
controls = cs.vertcat(
    omega_norm,
    Q_heat_norm
)
n_controls = controls.numel()

# disturbances 
current = cs.SX.sym("I_bat")
T_env = cs.SX.sym("T_env")
disturbances = cs.vertcat(current, T_env)
n_disturbances = disturbances.numel()

# matrix containing all states over all time steps
X = cs.SX.sym('X', n_states, N+1)

# matrix containing all control actions over all time steps
U = cs.SX.sym('U', n_controls, N)


# discretization model (e.g. x2 = f(x1, v, t) = x1 + v*dt)
# Parameters
m_battery = 20*2.5*4
c_battery = 795
c_coolant = 3500
density_coolant = 1050
omega = omega_scale * omega_norm
Q_heat = Q_heat_scale * Q_heat_norm
pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
mdot_c = density_coolant*pump_displacement*omega
R_battery = 4*20*0.0128 # Battery resistance
C_battery = 28*3600 # in coloumb
hA_bat = 2500
alpha_0 = 0.637968 #0.65
alpha_1 = 1 # 0.99 #0.998427 #0.99
alpha_2 = 0.980682  #0.999686 #.97
alpha_3 = 0.981036
gamma = 7.59853

# column vector for storing initial state, environment temp AND disturbances 
P = cs.SX.sym('P', n_states + n_disturbances)

# Get the cooler in and out temps
NTU_bat  = (alpha_3 * hA_bat) / (mdot_c*c_coolant + 1e-3)
T_clin= (T_bat + alpha_1*(1/(1-cs.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
T_clout = ((T_clin - T_bat) * alpha_2*cs.exp(-NTU_bat) + T_bat)
T_clin_min = CELSIUS_TO_KELVIN 
T_clin_max = 50 + CELSIUS_TO_KELVIN
T_clout_min = CELSIUS_TO_KELVIN 
T_clout_max = 50 + CELSIUS_TO_KELVIN
Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

# Now for the actual calculations 
T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma * (T_env - T_bat))
ode_model = T_bat_dot_model

f_model = cs.Function("f_model", [states,controls,disturbances], [ode_model], ["x", "u", "d"], ["ode"])
T_clin_fun = cs.Function("T_clin_fun", [states,controls], [T_clin], ["x", "u"], ["T_clin_fun"])
T_clout_fun = cs.Function("T_clout_fun", [states,controls], [T_clout], ["x", "u"], ["T_clout_fun"])

cost_fn = 0 # Cost function
g = X[:, 0] - P[:n_states] # constraints in the equation, that x0 = p0 

st = X[:, 0]
disturbances = P[-n_disturbances]

cost_fn = (U).T @ R @ (U)
T_clin_eval = T_clin_fun(st, U)
T_clout_eval = T_clout_fun(st, U)

st_next = X[:, 1]

K1 = f_model(st, U, disturbances)
K2 = f_model(st + step_horizon/2 * K1, U, disturbances)
K3 = f_model(st + step_horizon/2 * K2, U, disturbances)
K4 = f_model(st + step_horizon * K3, U, disturbances)

st_next_RK4 = st + (step_horizon) * (K1+ 2*K2 + 2*K3 + K4)

g = cs.vertcat(T_clin_eval, T_clout_eval, st - st_next, st_next - st_next_RK4) # Basically saying X[:, k+1] = runge kutta evaluation

OPT_variables = cs.vertcat(
    X.reshape((-1,1)),    # E.g. 3 x 11 --> 33 x 1
    U.reshape((-1,1)),     # E.g. 4 x 10 --> 40 x 1
)
nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
    'max_iter': 2000,
    'print_level': 0,
    'acceptable_tol': 1e-8,
    'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)

# Set constraints
omega_max = (4000*2*np.pi/60)/omega_scale
      
omega_min = (150*2*np.pi/60)/omega_scale
print("omega max", omega_max)
    
Q_heat_max = 4000/Q_heat_scale
print("Q_heat max", Q_heat_max)
Q_heat_min = 0/Q_heat_scale

lbx = cs.DM.zeros((n_states*(N+1) + N*n_controls, 1))
ubx = cs.DM.zeros((n_states*(N+1) + N*n_controls, 1)) # long column vector 

lbx[0: n_states*(N+1)] = T_bat_min # Different notation to matlab. Here have [start:stop.step]. We fill out the vector related to states [start:n_states*(N+1)] with a lower bound every n_states, meaning we set lower bound for first state at each time step
lbx[-2] = omega_min
lbx[-1] = Q_heat_min

ubx[0: n_states*(N+1)] = T_bat_max 
ubx[-2] = omega_max
ubx[-1] = Q_heat_max

lg = cs.DM.zeros(4, 1)
ug = cs.DM.zeros(4, 1)

lg[0] = T_clin_min # Odd is T_clin
ug[0] = T_clin_max
lg[1] = T_clout_min # Odd is T_clin
ug[1] = T_clout_max

args = {
    'lbg': lg, # constrainst lower bound (basically giving == constraint)
    'ubg':  ug,
    'lbx': lbx, 
    'ubx': ubx
}

X0 = cs.DM.ones((n_states * (N+1), 1))
u0 = cs.DM.ones((n_controls * N, 1))
state_init = 20.5 + CELSIUS_TO_KELVIN
T_env_0 = -10 + CELSIUS_TO_KELVIN
current_0 = 0
args['p'] = cs.vertcat(
    state_init, # current state
    T_env_0,
    current_0 
)
#print(args['p'])
# optimization variable current state
args['x0'] = cs.vertcat(
    cs.reshape(X0, n_states*(N+1), 1),
    cs.reshape(u0, n_controls*N, 1)
)

sol = solver(
    x0 = args['x0'],
    lbx = args['lbx'],
    ubx = args['ubx'],
    lbg = args['lbg'],
    ubg = args['ubg'],
    p = args['p']
)

print(sol['x'])