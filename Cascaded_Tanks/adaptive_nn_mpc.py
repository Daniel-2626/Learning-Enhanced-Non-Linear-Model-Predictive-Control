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
COST = "LINEAR_LS" # standard cost
SAVE_FLAG = False
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=3):
        super(MLP, self).__init__()
        self.tau = 2
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()] # nn.ReLU rectified linear function (max(x,0))
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]) # nn.Linear applies an affine transform. hidden_dim features and hidden_dim out features
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        net =  self.net(x)
        return self.tau* torch.tanh(net)
class CascadedTankLearnedDynamics:
    def __init__(self, residual_model): # remove gym_env for cascaded tank
        self.residual_model = residual_model # Residual model is L4Casadi residual
    
    def model(self):
        nominal_ratio = 0.7
        A1 = 1 #* nominal_ratio
        a1 = 0.1 * nominal_ratio
        A2 = 1 #* nominal_ratio
        a2 = 0.1 * nominal_ratio #* nominal_ratio
        
        k = 1000#*nominal_ratio
        rho = 1000 #*nominal_ratio
        
        g = 9.82
        
    
        # Input variables
        h1 = cs.MX.sym('h1')
        h2 = cs.MX.sym('h2')
        X = cs.vertcat(h1, h2)
        u = cs.MX.sym('u')
        #leakage = cs.MX.sym('leakage')
        nn_on = cs.MX.sym("nn_on")
        nx = 2
        nu = 1

        # Dynamics
        
        h1_dot =  k*u/(rho*A1) - a1/A1 * cs.sqrt(2*g*h1+0.00001)  #+ leakage
        h2_dot = a1/A1 * cs.sqrt(2*g*h1 + 0.00001) - a2/A2 * cs.sqrt(2*g*h2 + 0.00001) 
        X_dot_nominal = cs.vertcat(h1_dot, h2_dot)

        mlp_input = cs.vertcat(X, u)
        residual = self.residual_model(mlp_input.T).T 
        X_dot_residual = cs.vertcat(residual[0], residual[1]) # h1 dot and h2 dot residual

        f_expl = X_dot_nominal + nn_on * X_dot_residual # adding x dot residual leads to some stochasticity, maybe because we have a arbitrary neural network initially?
        x_start = np.array([1,1]) # initial position or initial guess?

        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = X
        model.xdot = cs.MX.sym('xdot', 2)
        model.u = u
        model.z = cs.vertcat([])
        model.p = nn_on #cs.vertcat([])  #leakage # cs.vertcat([])
        model.f_expl = f_expl
        model.f_nominal = X_dot_nominal
        model.x_start = x_start
        model.constraints = cs.vertcat([]) # add constraints here or in mpc?
        model.name = "cascaded_learned"
        
        return model 

class MPC:
    def __init__(self, model, N, t_horizon, external_shared_lib_dir, external_shared_lib_name):
        self.model = model
        self.N = N
        self.t_horizon = t_horizon
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
        ocp.cost.Vx = np.zeros((ny, nx)) # i feel like this should be nx,nx ||Vx x||^2 
        for i in range(nx):
            ocp.cost.Vx[i,i] = 1 # so set the matrix coeff corr to u to 0. maybe the dim of Vx is set to be the same as Vu
        ocp.cost.Vu = np.zeros((ny, nu))
        for i in range(nu):
            ocp.cost.Vu[i + nx, i] = 1
        ocp.cost.Vz = np.array([[]]) # don't know what the V_z z, what the variable z should be
        ocp.cost.Vx_e = np.eye(nx)

        ocp.parameter_values = 0
        l4c_y_expr = None

        # Define weight parameters
        Q = 1 * np.diag([10, 10])
        R = 1 * np.diag([0.1])
        ocp.cost.W = scipy.linalg.block_diag(Q,R)

        # Initial state (will be overwritten?)
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
        ocp.constraints.idxbx = np.array([0,1]) # at what indices to have constraints? 
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
        model_ac.f_impl_expr = model.xdot - model.f_expl # Implicit 0 = xdot - f
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.p = model.p
        model_ac.name = model.name
        return model_ac


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
## Actual model
#h1_dot = k*u/(rho*A1) - a1/A1 * cs.sqrt(2*g*h1+0.00001) #+ a2/A2 * cs.sqrt(2*g*h2 + 0.00001) 
h1_dot = k*u/(rho*A1)*cs.exp(-u/10) - a1/A1 * cs.sqrt(2*g*h1+0.00001) #+ a2/A2 * cs.sqrt(2*g*h2 + 0.00001) 

h2_dot = a1/A1 * cs.sqrt(2*g*h1 + 0.00001) - a2/A2 * cs.sqrt(2*g*h2 + 0.00001) #- 0.1*k*u/(rho*A2)
ode = cs.vertcat(h1_dot, h2_dot)
states = cs.vertcat(
    h1,
    h2
)
f = cs.Function("f", [states,u], [ode], ["x","u"], ["ode"])
        
def RK4(state, input_u, dt, f):
    K1 = f(state, input_u)
    K2 = f(state + dt/2 * K1, input_u)
    K3 = f(state + dt/2 * K2, input_u)
    K4 = f(state + dt * K3, input_u)
    next_state = state + (dt/6) * (K1 + 2*K2 + 2*K3 +K4)
    return next_state
# Residual MLP: Lightweight
residual_mlp = MLP(input_dim = 2 + 1, output_dim=2, hidden_dim=8, num_layers=1) # the network
residual_mlp.load_state_dict(torch.load("cascaded_tanks_pretrain.pth", weights_only=True))

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
print(nominal_func)
solver = MPC(model=learned_model.model(), N=N, t_horizon = t_horizon,
            external_shared_lib_dir=l4c_residual.shared_lib_dir,
            external_shared_lib_name=l4c_residual.name).solver # Returns the solver object from MPC

# Simulation setup 
dt = t_horizon/N
Tsim = 100
xt = np.array([0.5,0.5])
Steps = int(Tsim / dt)
h1_history, u_history, h1_ref_history, h2_ref_history, h2_history, opt_times = [xt[0]], [], [], [], [xt[1]], []

h1_ref = 1
h2_ref = 1
# Residual Finetune
obs_buffer = []
batch_size = 20
T_update = 20 # int(t_horizon//(dt))
nn_on = 0

residual_dictionary = {'run': [], 'h1':[], 'h2': [], 'u': [], 'residual_1': [], 'residual_2': []}
results_adaptive = {'run': [], 'h1':[], 'h2': [], 'u': []}

def DM2Arr(dm):
    # returns a full matrix instead if a soarse ibe
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

    #print(xt, ut)
    state_dynamics = DM2Arr(f(xt, ut))
    #print(state_dynamics)
    #stop
    mu = 0
    sigma = 0.01
    randn_1 = 0 #np.random.normal(mu,sigma)
    randn_2 = 0 #np.random.normal(mu, sigma)
    obs_buffer.append((xt[0], xt[1], ut, state_dynamics[0][0]+randn_1, state_dynamics[1][0]+randn_2))

    residual_1 = state_dynamics[0][0] - nominal_func(xt[:2], ut)[0][0]
    residual_2 = state_dynamics[1][0] - nominal_func(xt[:2], ut)[1][0]
    residual_dictionary['run'].append(0)
    residual_dictionary['h1'].append(xt[0])
    residual_dictionary['h2'].append(xt[1])
    residual_dictionary['u'].append(ut)
 
    residual_dictionary['residual_1'].append(residual_1)
    residual_dictionary['residual_2'].append(residual_2)
    #print(obs_buffer)
    print(i)
    # update every 50 time steps
    if i > 0 and (i % T_update) == 0 and len(obs_buffer) >= batch_size:
        nn_on = 1
        data = np.array(obs_buffer[-batch_size:])
        # data[:, :3] h1, h2 and u
        X_batch = torch.tensor(data[:, :3], dtype=torch.float32)
        # h1_dot, h2_dot
        #print(data[-1])
        y_true = data[:, 3:]
        #print(y_true)
       
        

        nominal = np.array([nominal_func(x[:2], x[2]).full().flatten() for x in data])
        #y_true = np.array([f(x[:2], x[3]).full().flatten() for x in data])
        y_nominal = nominal[:, :] # dx2 and dx4 from nominal model
        #print(y_nominal.shape)
        #print(y_nominal)
        #print(y_true.shape)
        #print(y_true)
        #print(y_true - y_nominal)
        #stop
        print(sum(y_true - y_nominal))
        y_target = torch.tensor(y_true - y_nominal, dtype=torch.float32)
        print(y_target)
        for p in residual_mlp.parameters(): p.requires_grad = True
        # An epoch
        for _ in range(50):
            residual_optimizer.zero_grad() # optimizer object
            prediction = residual_mlp(X_batch) # gives data to network to make a prediction
            loss = residual_criterion(prediction, y_target)
            #l2_norm = sum(p.pow(2).sum() for p in residual_mlp.parameters())
            #regularization = 0.1
            #loss += regularization * l2_norm
            loss.backward() # calculates gradient
            residual_optimizer.step() # one optimization step to update parameters
        for p in residual_mlp.parameters(): p.requires_grad = False
        l4c_residual.update(residual_mlp)

    elapsed= time.time() - start
    opt_times.append(elapsed)
# Input variables
h1 = cs.MX.sym('h1')
h2 = cs.MX.sym('h2')
X = cs.vertcat(h1, h2)
u = cs.MX.sym('u')
mlp_input = cs.vertcat(X, u)
residual = l4c_residual(mlp_input.T).T[0] 
f_res = cs.Function("f_res", [X,u], [residual])
weights = np.array([])
for p in residual_mlp.parameters():
    weights = np.append(weights, p.flatten())
print(weights)
print(len(weights))
print(residual.str(True))
state_target = np.array([h1_ref, h2_ref])
ss_error = cs.norm_2(xt - state_target)
print("final error", ss_error)
df = pd.DataFrame(data=residual_dictionary)
df.to_csv("cascaded_residuals.csv", index=False)

df = pd.DataFrame(data=results_adaptive)
df.to_csv("cascaded_adaptive_mismatched.csv", index=False)
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
