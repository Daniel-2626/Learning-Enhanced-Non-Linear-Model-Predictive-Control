import casadi as cs
import numpy as np
import torch
import torch.nn as nn
import l4casadi as l4c
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import time
import scipy.linalg
import matplotlib.pyplot as plt
import pandas as pd
import os 
from datetime import datetime
import random as random

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
### MLP ARCHITECTURE ###
# input_dim, output_dim, hidden_dim, num_layers will be changed lower in the code
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=3):
        super(MLP, self).__init__()
        self.tau = 2
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()] # nn.Tanh hyperbolic tangent
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]) # nn.Linear applies an affine transform
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        net =  self.net(x)
        # Bound NN output by tanh and confidence parameter tau
        return self.tau* torch.tanh(net)
### CASCADED TANKS DYNAMICS ###
class CascadedTankLearnedDynamics:
    def __init__(self, residual_model): 
        self.residual_model = residual_model # Residual model is L4Casadi residual
    
    def model(self):
        # Change for mistmach. Nominal = 1
        nominal_ratio = 0.7
        A1 = 1 
        a1 = 0.1 * nominal_ratio
        A2 = 1 
        a2 = 0.1 * nominal_ratio 
        
        k = 1000
        rho = 1000 
        g = 9.82
        
    
        # Input variables
        h1 = cs.MX.sym('h1')
        h2 = cs.MX.sym('h2')
        X = cs.vertcat(h1, h2)
        u = cs.MX.sym('u')
        
        # Switch to turn on neural network residual (gets turned on after initial training)
        nn_on = cs.MX.sym("nn_on")
        nx = 2
        nu = 1

        # Dynamics
        h1_dot =  k*u/(rho*A1) - a1/A1 * cs.sqrt(2*g*h1+0.00001) 
        h2_dot = a1/A1 * cs.sqrt(2*g*h1 + 0.00001) - a2/A2 * cs.sqrt(2*g*h2 + 0.00001) 
        X_dot_nominal = cs.vertcat(h1_dot, h2_dot)

        mlp_input = cs.vertcat(X, u)
        residual = self.residual_model(mlp_input.T).T 
        X_dot_residual = cs.vertcat(residual[0], residual[1]) # h1 dot and h2 dot residual

        f_expl = X_dot_nominal + nn_on * X_dot_residual 
        x_start = np.array([1,1]) # initial position, gets overwritten

        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = X
        model.xdot = cs.MX.sym('xdot', 2)
        model.u = u
        model.z = cs.vertcat([])
        model.p = nn_on 
        model.f_expl = f_expl
        model.f_nominal = X_dot_nominal
        model.x_start = x_start
        model.constraints = cs.vertcat([]) 
        model.name = "cascaded_learned"
        
        return model 
### MODEL PREDICTIVE CONTROLLER ### 
class MPC:
    def __init__(self, model, N, t_horizon, external_shared_lib_dir, external_shared_lib_name):
        self.model = model
        self.N = N
        self.t_horizon = t_horizon
        # Libraries for generated auto-generated jacobians/hessians by L4CasADi
        self.external_shared_lib_dir = external_shared_lib_dir 
        self.external_shared_lib_name = external_shared_lib_name

    @property # a decorator
    def solver(self):
        return AcadosOcpSolver(self.ocp())
    
    def ocp(self):
        model = self.model

        t_horizon = self.t_horizon
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model)

        # Dimensions
        nx = 2
        nu = 1
        ny = nx + nu # Stage cost
        ny_e = nx # Terminal cost considers only states

        # Create ocp to formulate optim
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.solver_options.tf = t_horizon

        # initialize cost function
        ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_e = 'LINEAR_LS'

        # state 
        ocp.cost.Vx = np.zeros((ny, nx))
        for i in range(nx):
            ocp.cost.Vx[i,i] = 1 
        ocp.cost.Vu = np.zeros((ny, nu))
        for i in range(nu):
            ocp.cost.Vu[i + nx, i] = 1

        ocp.cost.Vz = np.array([[]]) 
        ocp.cost.Vx_e = np.eye(nx)

        ocp.parameter_values = 0
        l4c_y_expr = None # Linear cost, no need to set y_expr in cost function

        # Define weight parameters
        Q = 1 * np.diag([10, 10])
        R = 1 * np.diag([0.1])
        ocp.cost.W = scipy.linalg.block_diag(Q,R)

        # Initial state 
        ocp.cost.W_e = Q 
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))
        
        # Initial state
        ocp.constraints.x0 = model.x_start

        # Set constraints
        u_max = 0.8
        h1_max = 2
        h2_max = 2
        ocp.constraints.lbu = np.array([0])
        ocp.constraints.ubu = np.array([u_max])
        ocp.constraints.idxbu = np.array([0])
        ocp.constraints.idxbx = np.array([0,1]) 
        ocp.constraints.ubx = np.array([h1_max, h2_max])
        ocp.constraints.lbx = np.array([0, 0])

        # Solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name 

        return ocp

    def acados_model(self, model):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.p = model.p
        model_ac.name = model.name
        return model_ac
### RK4 for simulating trajectory ### 
def RK4(state, input_u, dt, f):
    K1 = f(state, input_u)
    K2 = f(state + dt/2 * K1, input_u)
    K3 = f(state + dt/2 * K2, input_u)
    K4 = f(state + dt * K3, input_u)
    next_state = state + (dt/6) * (K1 + 2*K2 + 2*K3 +K4)
    return next_state
# -----------------------------------------------------------------------
# END OF CLASS/FUNCTION DEFINITIONS, BEGINNING OF SCRIPT
# -----------------------------------------------------------------------
### TRUE SIMULATION PARAMETERS 
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
## True plant model
#h1_dot = k*u/(rho*A1) - a1/A1 * cs.sqrt(2*g*h1+0.00001) #+ a2/A2 * cs.sqrt(2*g*h2 + 0.00001) 
# (Can introduce mismatch here, e.g. add terms to dynamics, see below for an example with nonlinear saturation (note u >= 0))
h1_dot = k*u/(rho*A1)*cs.exp(-u/10) - a1/A1 * cs.sqrt(2*g*h1+0.00001)

h2_dot = a1/A1 * cs.sqrt(2*g*h1 + 0.00001) - a2/A2 * cs.sqrt(2*g*h2 + 0.00001) #- 0.1*k*u/(rho*A2)

ode = cs.vertcat(h1_dot, h2_dot)
states = cs.vertcat(
    h1,
    h2
)
f = cs.Function("f", [states,u], [ode], ["x","u"], ["ode"])
        

# Residual MLP: Initialization
residual_mlp = MLP(input_dim = 2 + 1, output_dim=2, hidden_dim=8, num_layers=2) # the network
# Loaded MLP must have the same input_dim, output_dim, hidden_dim and num_layers
# Comment this line if do not want to use a pretrained network
residual_mlp.load_state_dict(torch.load("cascaded_tanks_deriv_pretrain.pth", weights_only=True))

# Freezes params in neural network, used when evaluating the network (e.g. in MPC optimization)
for param in residual_mlp.parameters():
    param.requires_grad = False

l4c_residual = l4c.L4CasADi(residual_mlp, name="cascadedtank", mutable=True)
residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=1e-2, weight_decay=0.3) # lr = learning rate, the optimizer
residual_criterion = nn.MSELoss()



# MPC Setup 
N = 50
t_horizon = 25
learned_model = CascadedTankLearnedDynamics(l4c_residual)
casadi_model = learned_model.model()

nominal_func = cs.Function('nom', [casadi_model.x, casadi_model.u], [casadi_model.f_nominal]) # x, u --> f What our controller knowns

solver = MPC(model=learned_model.model(), N=N, t_horizon = t_horizon,
            external_shared_lib_dir=l4c_residual.shared_lib_dir,
            external_shared_lib_name=l4c_residual.name).solver # Returns the solver object from MPC

# Simulation setup 
dt = t_horizon/N
Tsim = 100
xt = np.array([0.5,0.5]) # Initial state
Steps = int(Tsim / dt)
h1_history, u_history, h1_ref_history, h2_ref_history, h2_history, opt_times = [xt[0]], [], [], [], [xt[1]], []

# Reference
h1_ref = 1
h2_ref = 1

# Data for training residual
obs_buffer = []
batch_size = 50
T_update = 50 # int(t_horizon//(dt))
# Initially NN is off
nn_on = 0

# Used when saving to csv
residual_dictionary = {'run': [], 'h1':[], 'h2': [], 'u': [], 'residual_1': [], 'residual_2': []}
results_adaptive = {'run': [], 'h1':[], 'h2': [], 'u': []}

def DM2Arr(dm):
    # returns a full matrix instead if a sparse one
    return np.array(dm.full())

for i in range(Steps):
    current_time = i * dt

    # Set reference for each step in MPC horizon
    for k in range(N):
        if k == 0: # only store first reference
            h1_ref_history.append(h1_ref)
            h2_ref_history.append(h2_ref)
        # Set terminal reference
        y_ref_k = np.array([h1_ref, h2_ref, 0])
        solver.set(k, "yref", y_ref_k)
        solver.set(k, "p", nn_on)

    # Set terminal reference
    y_ref_terminal = np.array([h1_ref, h2_ref])
    solver.set(N, "yref", y_ref_terminal)
    solver.set(N, "p", nn_on)

    start = time.time()
    # Apply current state as constraint
    solver.set(0, "lbx", xt)
    solver.set(0, "ubx", xt)

    # Solve mpc and apply control
    solver.solve()
    ut = solver.get(0, "u").item()
    u_history.append(ut)


    results_adaptive['run'].append(0)
    results_adaptive['h1'].append(xt[0])
    results_adaptive['h2'].append(xt[1])
    results_adaptive['u'].append(ut)


    # Simulate forwards
    next_obs = RK4(xt, ut, dt, f)
    xt = DM2Arr(next_obs).ravel()
  
    h1_history.append(xt[0])
    h2_history.append(xt[1])

    # True state dynamics (from evaluating plant dynamics)
    state_dynamics = DM2Arr(f(xt, ut))

    # Observations consist of states, input, and state dynamics
    obs_buffer.append((xt[0], xt[1], ut, state_dynamics[0][0], state_dynamics[1][0]))
    # residual = plant dynamics - control model dynamics
    residual_1 = state_dynamics[0][0] - nominal_func(xt[:2], ut)[0][0]
    residual_2 = state_dynamics[1][0] - nominal_func(xt[:2], ut)[1][0]
    residual_dictionary['run'].append(0)
    residual_dictionary['h1'].append(xt[0])
    residual_dictionary['h2'].append(xt[1])
    residual_dictionary['u'].append(ut)
    residual_dictionary['residual_1'].append(residual_1)
    residual_dictionary['residual_2'].append(residual_2)

 
    # update every T_update iterations
    if i > 0 and (i % T_update) == 0 and len(obs_buffer) >= batch_size:
        # NN on after this step
        nn_on = 1
        data = np.array(obs_buffer[-batch_size:])
        # data[:, :3] = h1, h2, u
        X_batch = torch.tensor(data[:, :3], dtype=torch.float32)
        # h1_dot, h2_dot from plant dynamics
        y_true = data[:, 3:]
       
        nominal = np.array([nominal_func(x[:2], x[2]).full().flatten() for x in data])
        y_nominal = nominal[:, :] # h1_dot and h2_dot from control model

        # True - Nominal
        y_target = torch.tensor(y_true - y_nominal, dtype=torch.float32)
        
        for p in residual_mlp.parameters(): 
            p.requires_grad = True # Unfreezes params in neural network, used when training the network 

        # Epoch loop
        for _ in range(100):
            residual_optimizer.zero_grad() # optimizer object
            prediction = residual_mlp(X_batch) # gives data to network to make a prediction
            # Minimize (plant - (nominal + residual))^2
            loss = residual_criterion(prediction, y_target)

            loss.backward() 
            residual_optimizer.step() 
        for p in residual_mlp.parameters(): p.requires_grad = False
        l4c_residual.update(residual_mlp)

    elapsed= time.time() - start
    opt_times.append(elapsed)



state_target = np.array([h1_ref, h2_ref])
ss_error = cs.norm_2(xt - state_target)
print("final error", ss_error)

### SAVING RESULTS
#df = pd.DataFrame(data=residual_dictionary)
#df.to_csv("cascaded_residuals_fixed_epochs.csv", index=False)
df = pd.DataFrame(data=results_adaptive)
df.to_csv("cascaded_adaptive_deriv_mismatched.csv", index=False)

### PLOTTING
# Convert to numpy arrays for easier indexing
h1_history = np.array(h1_history)
h2_history = np.array(h2_history)
u_history = np.array(u_history)
h1_ref_history = np.array(h1_ref_history)
h2_ref_history = np.array(h2_ref_history)

# Create time grids with matching dimensions
t_grid_states = np.linspace(0, Tsim, len(h1_history))
t_grid_inputs = np.linspace(0, Tsim, len(u_history))

print(f'Mean iteration time: {1000*np.mean(opt_times):.1f}ms -- {1/np.mean(opt_times):.0f}Hz)')
print(f'State history shape: {h1_history.shape}, Control history shape: {u_history.shape}')

# ------------------------------------------------------------------------------
print("h final", h1_history[-1], h2_history[-1])
print(h1_history.shape)
# Plot
plt.figure(figsize=(15, 10))

# Plot x position
plt.subplot(3, 1, 1)
plt.plot(t_grid_states, h1_history[:], linewidth=2, color='C1', label='h1')
plt.plot(t_grid_inputs, h1_ref_history, '--', linewidth=2, label='h_1,ref', color='C0', alpha=0.7)
plt.ylabel('Position [m]')
plt.legend()
plt.grid()

# Plot theta angle
plt.subplot(3, 1, 2)
plt.plot(t_grid_states, h2_history[:], linewidth=2, color='C7', label='h2')
plt.plot(t_grid_inputs, h2_ref_history, '--', linewidth=2, label='h_2,ref', color='C0', alpha=0.7)
plt.ylabel('Angle [rad]')
plt.legend()
plt.grid()

# Plot control input
plt.subplot(3, 1, 3)
plt.plot(t_grid_inputs, u_history, linewidth=2, label='u')
plt.xlabel('Time [s]')
plt.ylabel('Control [N]')
plt.legend()
plt.grid()
plt.tight_layout()

plt.show()