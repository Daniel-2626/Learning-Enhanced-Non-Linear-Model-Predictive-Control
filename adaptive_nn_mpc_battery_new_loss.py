import casadi as cs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F # a bunch of functions, e.g. convolution

import l4casadi as l4c
from acados_template import AcadosOcpSolver, AcadosOcp, AcadosModel
import time
import scipy.linalg
import matplotlib.pyplot as plt
import pandas as pd
import os 
from datetime import datetime
import ctypes
import sys
import csv
import random as random
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from scipy.signal import savgol_filter


seed = sum([ord(char) for char in "DIAMONDS"])
np.random.seed(seed)
random.seed(seed)
CELSIUS_TO_KELVIN = 273.15
### MLP Network
class MLP(nn.Module):
    # Init values will overwritten in Controller class
    def __init__(self, input_dim=4, output_dim=2, hidden_dim=16, num_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        # Confidence parameter (c in thesis)
        self.tau = nn.Parameter(torch.tensor([0.001]), requires_grad=False)        
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        net = self.net(x)
        # Bounding network by tanh
        return self.tau * torch.tanh(net) 

### Model of battery dynamics, with residual nn, and optimization to find steady state
class BatteryLearnedDynamics:
    def __init__(self, residual_model): 
        self.residual_model = residual_model
    def model(self, T_env): 
        model = cs.types.SimpleNamespace()
        # Scaling on variables so they are in the same magnitude
        # Should be a min max scaling instead (see readme)
        model.omega_scale = 100
        model.Q_heat_scale = 1000
        model.T_bat_scale = 100
        model.current_scale = 25 

        model.T_env = T_env

        T_bat_normalized = cs.MX.sym('T_bat_normalized')
        T_bat = model.T_bat_scale * T_bat_normalized
        X = cs.vertcat(T_bat_normalized)
        
        # Pump angular velocity
        omega_normalized = cs.MX.sym('omega_norm')
        omega = model.omega_scale * omega_normalized 
        # Heating / cooling power into HCU
        Q_heat_normalized = cs.MX.sym('Q_heat_norm')
        Q_heat = model.Q_heat_scale * Q_heat_normalized
        # Current from drive cycle (will be given as a parameter) 
        current_normalized = cs.MX.sym('current_normalized')
        current = model.current_scale*current_normalized
        # Switch for turning nn_on (either 0 or 1). 
        # nn is off until first training
        nn_on = cs.MX.sym('nn_on') 

        P = cs.vertcat(current_normalized, nn_on)            
                
        U = cs.vertcat(omega_normalized, Q_heat_normalized)
        
        nx = 1
        nu = 2

        # Parameters
        m_battery = 20*2.5*4 # Mass of battery 
        c_battery = 795 # Specific heat capacity for battery
        c_coolant = 3500 # Specific heat capacity for coolant
        density_coolant = 1050
        
        pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
        R_battery = 4*20*0.0128 # Battery resistance
        C_battery = 28*3600 # Capacity
        hA_bat = 2500 # heat transfer constant

        
        # Identified fitting parameters
        alpha_0 = 0.635039  
        alpha_1 = 0.915692 
        alpha_2 = 0.919681 
        alpha_3 = 1.47275
        gamma = 7.38325 
       
        ## Dynamics
        # Mass flow rate
        mdot_c = density_coolant*pump_displacement*omega

        # Number of thermal units, coolant in and out temperatures. 1e-3 in denominator to avoid singularities
        NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
        T_clin= (T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
        T_clout = ((T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat)
        
        constraint = cs.types.SimpleNamespace()
        constraint.T_clin_min = -20 + CELSIUS_TO_KELVIN 
        constraint.T_clin_max = 100 + CELSIUS_TO_KELVIN
        constraint.T_clout_min = -20 + CELSIUS_TO_KELVIN
        constraint.T_clout_max = 100 + CELSIUS_TO_KELVIN
        constraint.expr = cs.vertcat(T_clin, T_clout)

        # Heating / cooling power transferred into battery from coolant
        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)
        # Dynamics
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))
        X_dot_nominal = cs.vertcat(T_bat_dot_model)
        T_bat_dot_function = cs.Function("f_model", [T_bat_normalized, current_normalized, omega_normalized, Q_heat_normalized], 
        [T_bat_dot_model], ["T_bat", "current", "omega_normalized", "Q_heat_normalized"], ["ode"])

        # MLP network
        # Inputs: T_bat, current, omega, Q_heat (scaled versions)
        mlp_input = cs.vertcat(T_bat_normalized, current_normalized, omega_normalized, Q_heat_normalized) 
        residual = self.residual_model(mlp_input.T).T 

        X_dot_residual = cs.vertcat(residual[0]) # residual network
        # nominal model + residual model 
        f_expl = (X_dot_nominal + nn_on*X_dot_residual)/model.T_bat_scale
        x_start = np.array([T_env/model.T_bat_scale]) # initial constraint (gets overwritten)

        # store to struct
        
        model.x = X 
        model.xdot = cs.MX.sym('xdot', 1)
        model.u = U
        model.z = cs.vertcat([])
        model.p = P
        model.f_expl = f_expl
        model.f_nominal = X_dot_nominal
        model.x_start = x_start
        model.constraints = cs.vertcat([])
        model.name = "battery"
        model.T_bat_dot_function = T_bat_dot_function
        
        return model, constraint 
    def optimization_problem_steady_state(self, dt, T_env):
        ## Steady state selection optimization problem
        T_bat_scale = 100
        omega_scale = 100
        current_scale = 25
        Q_heat_scale = 1000

        # Add scaling to cost so cost is on unscaled T_bat

        # Works in normalized 
        Q = np.diag([10*T_bat_scale**2])
        R = np.diag([1, 10])   
        # Cost for slack
        T = np.diag([10000000,10000000])


        N = 1 # number of look ahead steps

        t_env = T_env
        step_horizon = dt # time between steps in seconds

        # Constraints
        T_bat_max = 50 + CELSIUS_TO_KELVIN
        T_bat_min = -20 + CELSIUS_TO_KELVIN

        # State symbolic variables
        T_bat_normalized = cs.SX.sym('T_bat_normalized')
        
        T_bat = T_bat_normalized*T_bat_scale        
        slack_RK4 = cs.SX.sym("Slack RK4")
        slack_ss = cs.SX.sym("Slack ss")
        S = cs.vertcat(slack_ss, slack_RK4)
        n_slack = S.numel()
        states = cs.vertcat(
            T_bat_normalized,
        )
        n_states = states.numel() # Returns amount of elements

        # control symbolic variables
        omega_norm = cs.SX.sym('omega_norm')
        Q_heat_norm = cs.SX.sym('Q_heat_norm')
        controls = cs.vertcat(
            omega_norm,
            Q_heat_norm
        )
        n_controls = controls.numel()
        
        # disturbances 
        current_normalized = cs.SX.sym("I_bat_norm")
        current = current_scale*current_normalized

        disturbances = cs.vertcat(current_normalized)
        n_disturbances = disturbances.numel()

        # matrix containing all states over all time steps
        X = cs.SX.sym('X', n_states, N+1)

        # matrix containing all control actions over all time steps
        U = cs.SX.sym('U', n_controls, N)

        T_env = T_env
        T_bat_target = (20.5 + CELSIUS_TO_KELVIN)/T_bat_scale

        # Model parameters (same as in BatteryLearnedDynamics)
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
        # Fitting parameters (same as in BatteryLearnedDynamics)
        alpha_0 = 0.635039 
        alpha_1 = 0.915692 
        alpha_2 = 0.919681 
        alpha_3 = 1.47275 
        gamma = 7.38325  

        # column vector for storing disturbances 
        P = cs.SX.sym('P', n_disturbances)

        # Dynamics (same as in BatteryLearnedDynamics)

        NTU_bat  = (alpha_3 * hA_bat) / (mdot_c*c_coolant + 1e-3)
        T_clin= (T_bat + alpha_1*(1/(1-cs.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
        T_clout = ((T_clin - T_bat) * alpha_2*cs.exp(-NTU_bat) + T_bat)
        T_clin_min = -20 + CELSIUS_TO_KELVIN
        T_clin_max = 100 + CELSIUS_TO_KELVIN # # Used to be 50
        T_clout_min = -20 + CELSIUS_TO_KELVIN 
        T_clout_max = 100 + CELSIUS_TO_KELVIN
        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)
        
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma * (T_env - T_bat))
        ode_model = T_bat_dot_model

        f_model = cs.Function("f_model", [states,controls,disturbances], [ode_model], ["x", "u", "d"], ["ode"])
      
        T_clin_fun = cs.Function("T_clin_fun", [states,controls], [T_clin], ["x", "u"], ["T_clin_fun"])
        T_clout_fun = cs.Function("T_clout_fun", [states,controls], [T_clout], ["x", "u"], ["T_clout_fun"])

        cost_fn = 0 # Cost function
        g = X[:, 0] - T_bat_target # constraints in the equation, that x0 = p0 

        st = X[:, 0]
        disturbances = P[-n_disturbances:]

        cost_fn = (U).T @ R @ (U) + S.T @ T @ S + (X[:, 1]-X[:,0]).T @ Q @ (X[:, 1] -X[:,0])
        T_clin_eval = T_clin_fun(st, U)
        T_clout_eval = T_clout_fun(st, U)

        st_next = X[:, 1]
        K1 = f_model(st, U, disturbances[0]) 
        # Not using NN in steady state selection problem, just nominal dynamics
        st_next_RK4 = st + (step_horizon) * K1 
        

        g = cs.vertcat(g,T_clin_eval, T_clout_eval, st_next - st_next_RK4 + slack_RK4) # Basically saying X[:, k+1] = runge kutta evaluation

        OPT_variables = cs.vertcat(
            X.reshape((-1,1)),    # E.g. 3 x 11 --> 33 x 1
            U.reshape((-1,1)),     # E.g. 4 x 10 --> 40 x 1
            S.reshape((-1,1))
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
            'acceptable_obj_change_tol': 1e-6,
            'hessian_approximation': 'limited-memory'
        },
            'print_time': 0
        }
   
        steady_state_solver = cs.nlpsol('solver', 'ipopt', nlp_prob, opts)
        # Set constraints
        omega_max = (4000*2*np.pi/60)/omega_scale
            
        omega_min = (150*2*np.pi/60)/omega_scale
            
        Q_heat_max = 4000/Q_heat_scale
        Q_heat_min = -4000/Q_heat_scale

        lbx = cs.DM.zeros((n_states*(N+1) + N*n_controls + N*n_slack, 1))
        ubx = cs.DM.zeros((n_states*(N+1) + N*n_controls + N*n_slack, 1)) # long column vector 

        lbx[0: n_states*(N+1)] = T_bat_min/T_bat_scale # Different notation to matlab. Here have [start:stop.step]. We fill out the vector related to states [start:n_states*(N+1)] with a lower bound every n_states, meaning we set lower bound for first state at each time step
        lbx[-4] = omega_min
        lbx[-3] = Q_heat_min
        lbx[-2:] = -100
        ubx[0: n_states*(N+1)] = T_bat_max/T_bat_scale
        ubx[-4] = omega_max
        ubx[-3] = Q_heat_max
        ubx[-2:] = 100
        lg = cs.DM.zeros(4, 1)
        ug = cs.DM.zeros(4, 1)


        lg[1] = T_clin_min # Odd is T_clin
        ug[1] = T_clin_max
        lg[2] = T_clout_min # Odd is T_clin
        ug[2] = T_clout_max
        X0 = cs.DM.ones((n_states * (N+1), 1))
        u0 = cs.DM.ones((n_controls * N, 1))
        s0 = cs.DM.zeros((n_slack*N, 1))
        steady_state_args = {
            'lbg': lg, # constrains lower bound (basically giving == constraint)
            'ubg':  ug,
            'lbx': lbx, 
            'ubx': ubx
        }
             # optimization variable current state
        steady_state_args['x0'] = cs.vertcat(
            cs.reshape(X0, n_states*(N+1), 1),
            cs.reshape(u0, n_controls*N, 1),
            cs.reshape(s0, n_slack * N ,1)
        )
        return steady_state_solver, steady_state_args

### Model predictive controller definition
class MPC:
    def __init__(self, model, constraint, N, t_horizon, external_shared_lib_dir, external_shared_lib_name):
        # Model and constraints from BatteryLearnedDynamics
        self.model = model
        self.constraint = constraint
        self.N = N
        self.t_horizon = t_horizon
        # For generated code by L4CasADi
        self.external_shared_lib_dir = external_shared_lib_dir
        self.external_shared_lib_name = external_shared_lib_name

    @property 
    def solver(self):
        return AcadosOcpSolver(self.ocp())
    
    def ocp(self):
        model = self.model
        constraint = self.constraint

        t_horizon = self.t_horizon
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model, constraint=constraint)
        # Expressions for non linear constraints (coolant temperatures)
        model_ac.con_h_expr = constraint.expr
        model_ac.con_h_expr_0 = constraint.expr
        # Dimensions
        nx = 1
        nu = 2
        ny = nx + nu # Stage cost
        ny_e = nx # Terminal cost considers only states
        nh = constraint.expr.shape[0]
        
        nsh = nh
        nsu = nu
        nsx = 1
        ns = nsu + nsx + nsh

        # Create ocp to formulate optim
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.dims.nh = nh
        ocp.dims.ns = ns
        ocp.dims.nsbx = nsx
        ocp.dims.nsbu = nsu

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
        l4c_y_expr = None

        # Define weight parameters
        Q = np.diag([10*model.T_bat_scale**2])
        R = np.diag([1, 10])
        ocp.cost.W = scipy.linalg.block_diag(Q,R)
        ocp.cost.W_e = Q 
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # Initial state
        ocp.constraints.x0 = model.x_start

        # Set constraints
        omega_max = (4000*2*np.pi/60)/model.omega_scale
        omega_min = (150*2*np.pi/60)/model.omega_scale    
        Q_heat_max = 4000/model.Q_heat_scale
        Q_heat_min = -4000/model.Q_heat_scale

        Tb_max = (50 + CELSIUS_TO_KELVIN)/model.T_bat_scale
        Tb_min = (-20 + CELSIUS_TO_KELVIN)/model.T_bat_scale

        ocp.constraints.lbu = np.array([omega_min, Q_heat_min])
        ocp.constraints.ubu = np.array([omega_max, Q_heat_max])

        ocp.constraints.idxbu = np.array([0,1])
        ocp.constraints.idxbx = np.array([0]) 
        ocp.constraints.ubx = np.array([Tb_max])
        ocp.constraints.lbx = np.array([Tb_min])
        

        ocp.cost.zl = 100000 * np.zeros((ns,))
        ocp.cost.zu = 100000 * np.zeros((ns,))
        ocp.cost.Zl = 100000* np.ones((ns,))
        ocp.cost.Zu = 100000 * np.ones((ns,))

        ocp.cost.zl_0 = 100000 * np.zeros((nsh+nsu,))
        ocp.cost.zu_0 = 100000 * np.zeros((nsh+nsu,))
        ocp.cost.Zl_0 = 100000* np.ones((nsh+nsu,))
        ocp.cost.Zu_0 = 100000 * np.ones((nsh+nsu,))

        ocp.constraints.lh = np.array(
        [
            constraint.T_clin_min,
            constraint.T_clout_min
        ]
        )
        ocp.constraints.uh = np.array(
            [
                constraint.T_clin_max,
                constraint.T_clout_max
            ]
        )

        ocp.constraints.lh_0 = np.array(
        [
            constraint.T_clin_min,
            constraint.T_clout_min
        ]
        )
        ocp.constraints.uh_0 = np.array(
            [
                constraint.T_clin_max,
                constraint.T_clout_max
            ]
        )

        ocp.constraints.idxsbx = np.array(range(nsx))
        ocp.constraints.idxsbu = np.array(range(nsu))
        ocp.constraints.idxsh = np.array(range(nsh))
        ocp.constraints.idxsh_0 = np.array(range(nsh))
        
        # Solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        # Libraries (folders) for auto-generated L4CasADi code, e.g. jacobians
        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name 
        ocp.solver_options.sim_method_num_stages = 1 # Use explicit euler integration
        
        # Will be overwritten
        ocp.parameter_values = np.zeros((2,))

        return ocp

    def acados_model(self, model, constraint):
        model_ac = AcadosModel()
        model_ac.f_impl_expr = model.xdot - model.f_expl # Implicit 0 = xdot - f
        model_ac.f_expl_expr = model.f_expl
        model_ac.x = model.x
        model_ac.xdot = model.xdot
        model_ac.u = model.u
        model_ac.p = model.p
        model_ac.name = model.name
        model_ac.con_h_expr = constraint.expr
        return model_ac


### Controller including setup, get_steady_state, get_cost_to_go, collect_data, get_input
class Controller:
    def setup(self, T_bat_target, T_env):
        # MPC Setup 
        self.N = 200 
        self.t_horizon = self.N*5
        self.dt = self.t_horizon/self.N

        # NN architecture and hyperparameter
        learning_rate = 1e-2
        input_dim = 4
        output_dim = 1
        hidden_dim = 32
        num_layers = 2 # amount of hidden layers
        weight_decay = 0.1

        self.lambda_jac = 0.1 # Weight of jacobian loss
        self.epsilon = 0.1 # Noise added in training 
        self.batch_size = 200 # how much data to collect for NN training
        self.T_update = self.batch_size*self.dt # How often NN is trained
        self.T_rollout = 20 # Rollout length (how often re-initialize true temperature in trajectory simulation)
        self.noise_std = 0.005 # Noise added to prediction to make network more robust
        self.confidence = 0.001 


        # Residual MLP
        residual_mlp = MLP(input_dim = input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers) # the network

        # Freeze parameters        
        for param in residual_mlp.parameters():
            param.requires_grad = False
        residual_mlp.load_state_dict(torch.load("Pretrained_Networks/heating_pretrain_scaled_network_4_input_new_loss_filtered_tanh_jac_penalty_phys_penalty_FINAL_FINAL.pth", weights_only=True))

        self.residual_mlp = residual_mlp
        self.residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay) 
        self.residual_criterion = nn.MSELoss()

        l4c_residual = l4c.L4CasADi(self.residual_mlp, name="battery", mutable=True)
        self.l4c_residual = l4c_residual

        learned_model = BatteryLearnedDynamics(l4c_residual)
        casadi_model, constraint = learned_model.model(T_env=T_env)

        self.T_bat_dot_function = casadi_model.T_bat_dot_function

        self.solver = MPC(model=casadi_model, constraint=constraint, N=self.N, t_horizon = self.t_horizon,
                    external_shared_lib_dir=l4c_residual.shared_lib_dir,
                    external_shared_lib_name=l4c_residual.name).solver # Returns the solver object from MPC

        self.obs_buffer = []
                
        # Time for which simulation is warm started. I.e. for the first 30 s a pre-determined input is used
        self.T_warm_start = 30 # Helps with initializing the Simulink so that the first iterations are not super slow

        self.current_iterate = 0
        self.xt_pred = np.array([CELSIUS_TO_KELVIN])

        self.x_last = np.array([0])
        self.omega_last = 0
        self.Q_heat_last = 0
        self.current_last = 0
        self.T_bat_last = 0
        
        self.data = []
        self.dynamics = []
        self.total_errors = 0
        self.total_cost = 0
        self.omega_scale = casadi_model.omega_scale
        self.Q_heat_scale = casadi_model.Q_heat_scale
        self.T_bat_scale = casadi_model.T_bat_scale
        self.current_scale = casadi_model.current_scale
        self.residual_scale = casadi_model.residual_scale
        self.T_env = T_env
        # Switch for NN, gets turned on after initial training
        self.nn_on = 0

        # Reading disturbance info
        df = pd.read_csv("current_rms_5s.csv", names=["current"])
        arr = df.to_numpy(dtype=np.float32)
        self.disturbance_values = arr

        self.steady_state_solver, self.steady_state_args = learned_model.optimization_problem_steady_state(dt=self.dt, T_env=T_env)
        
    def get_steady_state(self, T_bat_target):
        # Gets steady state for tracking (so reference on input is not 0, but u_ss)
        args = self.steady_state_args
        
        disturbances = self.disturbance_values
        current_N = disturbances[self.current_iterate + self.N].item()/self.current_scale
        args['p'] = cs.vertcat(
            current_N
        )
        # Call optimization problem for steady state (defined in LearnedBatteryDynamics)
        sol = self.steady_state_solver(
            x0 = args['x0'],
            lbx = args['lbx'],
            ubx = args['ubx'],
            lbg = args['lbg'],
            ubg = args['ubg'],
            p = args['p']
        )
        omega_norm_ss = sol['x'][2]
        Q_heat_norm_ss = sol['x'][3]
        print('solved steady state', sol['x'])

        return omega_norm_ss.full().item(), Q_heat_norm_ss.full().item()
    # Calculates cost-to-go for recursive feasibility (linearized)
    def get_cost_to_go(self, T_bat_target, T_env, omega_norm_ss, Q_heat_norm_ss):
        T_bat_target = T_bat_target/self.T_bat_scale
        T_bat_normalized = cs.MX.sym('T_bat_normalized')
        T_bat = self.T_bat_scale * T_bat_normalized
        omega_norm = cs.MX.sym('omega_norm')
        omega = self.omega_scale * omega_norm
        Q_heat_norm = cs.MX.sym('Q_heat_norm')
        Q_heat = self.Q_heat_scale * Q_heat_norm 

        U = cs.vertcat(omega_norm, Q_heat_norm)

        current_normalized = cs.MX.sym("curent_normalized")
        current = self.current_scale*current_normalized

        disturbances = self.disturbance_values

        current_N = disturbances[self.current_iterate + self.N].item()/self.current_scale

        # Parameters (same as in BatteryLearnedDynamics)
        m_battery = 20*2.5*4
        c_battery = 795
        c_coolant = 3500
        density_coolant = 1050
                
        pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
        R_battery = 4*20*0.0128 # Battery resistance
        C_battery = 28*3600 # in coloumb
        hA_bat = 2500        

        # Fitting parameters (same as in BatteryLearnedDynamics)
        alpha_0 = 0.635039 
        alpha_1 = 0.915692
        alpha_2 = 0.919681 
        alpha_3 = 1.47275 
        gamma = 7.38325 

        # Dynamics
        mdot_c = density_coolant*pump_displacement*omega
       
        NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
        T_clin= (T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
        T_clout = ((T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat)
        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))
        

        # Linearization by calculating Jacobians
        f_T_bat_dot = cs.Function("f_model", [T_bat_normalized,current_normalized, U], [T_bat_dot_model], ["x", "I_bat", "u"], ["ode"])

        jac_T_bat = cs.jacobian(T_bat_dot_model, T_bat_normalized)
        jac_T_bat_fun = cs.Function("f_dT_bat", [T_bat_normalized,current_normalized, U], [jac_T_bat], ["x", "I_bat", "u"], ["f_dT_bat"])

        jac_U = cs.jacobian(T_bat_dot_model, U)
        jac_U_fun = cs.Function("f_dU", [T_bat_normalized, current_normalized, U], [jac_U], ["x", "I_bat", "u"], ["f_dU"])

        A = jac_T_bat_fun(T_bat_target, current_N, [omega_norm_ss, Q_heat_norm_ss]).full()
        B = jac_U_fun(T_bat_target, current_N, [omega_norm_ss, Q_heat_norm_ss]).full()
        B = B.reshape((1,2))

        a = A
        b = B
        q = 10*self.T_bat_scale**2
        r = np.diag([1,10])
        cost_to_go = scipy.linalg.solve_continuous_are(a = a, b = b, q = q, r = r).item()
        Q_e = np.diag([cost_to_go])
        return Q_e, cost_to_go
    def get_input(self, T_bat_target, T_bat_0, SOC_0, current_0, dT_bat, dSOC):
        # MPC step that returns input (and other values for plotting)
        print("--------------------------------")
        print("ADAPTIVE TRAJECTORY LOSS MPC BATTERY MODEL WITH TARGET TRACKING AND COST-TO-GO")
        start = time.time()
        T_bat_target = T_bat_target/self.T_bat_scale

        if self.current_iterate == 0:
            self.xt_pred = ([T_bat_0])       
            # Warm start solver for iterate = 0
            omega_warm_start = (2000*2*np.pi/60)/self.omega_scale
            if T_bat_0 <= 20.5:
                Q_heat_warm_start = 2000/self.Q_heat_scale
            else:
                Q_heat_warm_start = -2000/self.Q_heat_scale
            input_warm_start = np.array([omega_warm_start, Q_heat_warm_start])
            self.solver.set(0, 'u', input_warm_start) 
            state_warm_start = np.array([T_bat_0/self.T_bat_scale])
            self.solver.set(0, 'x', state_warm_start)
        disturbances = self.disturbance_values
        T_env = self.T_env

        # Get steady state reference for inputs
        omega_norm_ss, Q_heat_norm_ss = self.get_steady_state(T_bat_target=T_bat_target)
        # Get cost-to-go weighting matrix
        Q_e, cost_to_go = self.get_cost_to_go(T_bat_target, T_env, omega_norm_ss, Q_heat_norm_ss)
        
        # Set reference and current (disturbance) for each step in MPC horizon
        for k in range(self.N):
            y_ref_k = np.array([T_bat_target, omega_norm_ss, Q_heat_norm_ss])
         
            self.solver.set(k, "yref", y_ref_k)
            param_values = np.array([disturbances[self.current_iterate + k].item()/self.current_scale, self.nn_on])

            self.solver.set(k, "p", param_values)

        # Set terminal reference
        y_ref_terminal = np.array([T_bat_target]) # Terminal cost only on state

        self.solver.set(self.N, "yref", y_ref_terminal)
        # Set current on last time step
        param_values = np.array([disturbances[self.current_iterate + self.N].item()/self.current_scale, self.nn_on])

        self.solver.set(self.N, "p", param_values)

        # Set cost-to-go on terminal node
        self.solver.cost_set(self.N, 'W', Q_e)

        xt = np.array([T_bat_0/self.T_bat_scale])
        
        # Apply current state as constraint
        self.solver.set(0, "lbx", xt)
        self.solver.set(0, "ubx", xt)

        omega_min, omega_max = (150*2*np.pi/60)/self.omega_scale, (4000*2*np.pi/60)/self.omega_scale
        Q_heat_min, Q_heat_max = -4000/self.Q_heat_scale, 4000/self.Q_heat_scale
       
        # Solve mpc and apply control
        self.solver.solve()
        self.total_cost += self.solver.get_cost()
        ut = self.solver.get(0, "u")

        shooting_node = 1
        slack_lower = self.solver.get(shooting_node, "sl")
        slack_upper = self.solver.get(shooting_node, "su")
        slack_x = [slack_upper[2], slack_lower[2]]
        print("Slacks for shooting node {}".format(shooting_node))
        print("slack lower", slack_lower, "slack upper", slack_upper)
        # Saturation
        omega_value = min(ut[0].item(), omega_max)
        Q_heat_value = min(ut[1].item(), Q_heat_max)
        omega_value = max(omega_value, omega_min)
        Q_heat_value = max(Q_heat_value, Q_heat_min)
        omega_value = self.omega_scale * omega_value
        Q_heat_value = self.Q_heat_scale * Q_heat_value
        status = self.solver.get_status()
        if status != 0:
            print("ERROR", status, "at iterate", self.current_iterate)
            self.total_errors += 1
        
        pred_error = 0
        pred_error_nn = 0
        T_bat_pred = 0
        T_bat_pred_nn = 0
        dT_bat_euler = 0
        if self.current_iterate  >= 0:
            # Collect training data for NN
            omega = self.omega_last
            Q_heat = self.Q_heat_last 

            current = self.current_last
            T_bat = self.T_bat_last
            dT_bat_euler = (T_bat_0 - T_bat)/self.dt

            # Calculate derivative with model
            # Parameters (same as in BatteryLearnedDynamics)
            m_battery = 20*2.5*4
            c_battery = 795
            c_coolant = 3500
            density_coolant = 1050
            pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
            R_battery = 4*20*0.0128 # Battery resistance
            C_battery = 28*3600 # in coloumb
            hA_bat = 2500

            # Fitting parameters (same as in BatteryLearnedDynamics)
            alpha_0 = 0.635039 
            alpha_1 = 0.915692 
            alpha_2 = 0.919681  
            alpha_3 = 1.47275
            gamma = 7.38325
            
            # Dynamics
            mdot_c = density_coolant*pump_displacement*omega
            NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
            T_clin= (T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
            T_clout = ((T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat)
          
            Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

            T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(self.T_env - T_bat))
            self.dynamics.append(T_bat_dot_model)
            self.data.append((T_bat/self.T_bat_scale, current/self.current_scale, omega/self.omega_scale, Q_heat/self.Q_heat_scale))
            
            
            self.residual_mlp.eval()
            # Training data, all inputs to NN (T_bat, current, omega, Q_heat)
            residual_data = torch.tensor([T_bat/self.T_bat_scale, current/self.current_scale, omega/self.omega_scale, Q_heat/self.Q_heat_scale], dtype=torch.float32)
            
            residual = self.residual_mlp(residual_data).numpy().item()
            T_bat_dot_nn = T_bat_dot_model + self.nn_on*residual/self.residual_scale
            
            T_bat_pred = T_bat + self.dt*T_bat_dot_model
            T_bat_pred_nn = T_bat + self.dt*T_bat_dot_nn
            pred_error_nn = T_bat_0 - T_bat_pred_nn 
            pred_error = T_bat_0 - T_bat_pred   
            pred_error_dt = pred_error/self.dt
            self.residual_mlp.train()

        # NN training
        # Occurs after a batch of batch size has been collected, and every T_update, and only after warm start has been finished
        if (self.dt*self.current_iterate) > self.T_warm_start and ((self.dt*self.current_iterate) % self.T_update) == 0 and len(self.data) >= self.batch_size:

            self.nn_on = 1 # NN_on switch
            data = np.array(self.data[-self.batch_size:])
            m_battery = 20*2.5*4
            c_battery = 795
            c_coolant = 3500
            density_coolant = 1050
                            
            pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
            R_battery = 4*20*0.0128 # Battery resistance
            C_battery = 28*3600 # in coloumb
            hA_bat = 2500
            alpha_0 = 0.635039 
            alpha_1 = 0.915692 
            alpha_2 = 0.919681  
            alpha_3 = 1.47275
            gamma = 7.38325
            dt = self.dt
            T_env = self.T_env

            # Unfreeze parameters for training
            for p in self.residual_mlp.parameters(): p.requires_grad = True
            ## Filtering
            polyorder = 2
            window_length = 10
            temperatures_data = data[:,0].flatten()
            temperatures_filtered = savgol_filter(temperatures_data, window_length=window_length, polyorder=polyorder)
            temperatures_filtered = temperatures_filtered.reshape((-1,1))

            current_data = data[:,1].flatten()
            current_filtered = savgol_filter(current_data, window_length=window_length, polyorder=polyorder)
            current_filtered = current_filtered.reshape((-1,1))
            omega_scaled_data = data[:,2].flatten()
            omega_scaled_filtered = savgol_filter(omega_scaled_data, window_length=window_length, polyorder=polyorder)
            omega_scaled_filtered = omega_scaled_filtered.reshape((-1,1))
            Q_heat_scaled_data = data[:,3].flatten()
            Q_heat_scaled_filtered = savgol_filter(Q_heat_scaled_data, window_length=window_length, polyorder=polyorder)
            Q_heat_scaled_filtered = Q_heat_scaled_filtered.reshape((-1,1))


            # Calculate confidence in nominal model (not used here, confidence is obtained by hardcoded)
            # But start for an automatic confidence calculation 
            # Need to filter first, i.e. need to recalculate dynamics
            dynamics_model = []
            for k in range(self.batch_size):
                omega_scaled = omega_scaled_filtered[k]
                omega = self.omega_scale*omega_scaled
                Q_heat_scaled = Q_heat_scaled_filtered[k]
                Q_heat = self.Q_heat_scale*Q_heat_scaled
                                    
                current_scaled = current_filtered[k]            
                current = self.current_scale*current_scaled
                        
                T_bat = self.T_bat_scale* temperatures_filtered[k]
                # Dynamics
                mdot_c = density_coolant*pump_displacement*omega
                               
                NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
                T_clin= (T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
                T_clout = ((T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat)
                        
                Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

                T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))
                dynamics_model.append(T_bat_dot_model)
                    
            dynamics_savgol = self.T_bat_scale*savgol_filter(temperatures_data, window_length=window_length, polyorder=polyorder, deriv = 1, delta = self.dt)
            confidence = self.confidence
            # Uncomment if want to use calculated confidence, and change it automatically
            #np.sqrt(np.power(dynamics_savgol-dynamics_model,2).mean())
            with torch.no_grad():
                self.residual_mlp.tau.copy_(torch.tensor([confidence]))            
            print("confidence", confidence)
            # Use filtered temperatures and current here, could consider unfiltered
            temperatures_tensor = torch.tensor(temperatures_filtered, dtype=torch.float32)
            current_tensor = torch.tensor(current_filtered,  dtype=torch.float32)
            omega_scaled_tensor = torch.tensor(omega_scaled_data, dtype=torch.float32)
            Q_heat_scaled_tensor = torch.tensor(Q_heat_scaled_data, dtype=torch.float32)
         
            T_rollout = self.T_rollout
            epsilon = self.epsilon
            noise_std = self.noise_std

            # Epoch loop
            for epoch in range(200):
                self.residual_optimizer.zero_grad() 
                loss = 0
                traj_loss = 0
                jacobian_loss = 0
                residual_loss = 0
                T_bat = self.T_bat_scale*temperatures_tensor[0]
                weight = 1.02

                # Rollout loop. Predict for T_rollout steps, then re-initialize with true temperature
                for k0 in range(0,self.batch_size-1, T_rollout):
                    T_bat =  self.T_bat_scale*temperatures_tensor[k0]
                    
                    for i in range(T_rollout):

                        t = k0 + i
                        if (t+1) >= self.batch_size:
                            break
                        omega_scaled = omega_scaled_tensor[t]
                        omega = self.omega_scale*omega_scaled
                        Q_heat_scaled = Q_heat_scaled_tensor[t]
                        Q_heat = self.Q_heat_scale*Q_heat_scaled
                                    
                        current_scaled = current_tensor[t]            
                        current = self.current_scale*current_scaled
                        
                        T_bat_next = self.T_bat_scale* temperatures_tensor[t+1]
                        # Added noise to prediction to teach network not to overcorrect
                        T_bat_noisy = T_bat + torch.randn_like(T_bat) * noise_std
                        
                        
                        # Dynamics

                        mdot_c = density_coolant*pump_displacement*omega 
                        NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
                        # Clamping NTU to avoid singularities or exploding values
                        NTU_safe = torch.clamp(NTU_bat, min=1e-6, max=50) 

                        denom = 1 - torch.exp(-NTU_safe)
                        # To ensure denom is not 0 (singularity)
                        safe_denom = torch.clamp(denom, min=1e-6)

                        T_clin= (T_bat_noisy + alpha_1*(1/(safe_denom))*Q_heat/(mdot_c*c_coolant + 1e-3))
                        T_clout = ((T_clin - T_bat_noisy) * alpha_2*np.exp(-NTU_safe) + T_bat_noisy)
                            
                        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

                        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat_noisy))

                        if torch.isnan(T_bat_dot_model).any():
                            print(f"NaN in Physics! NTU: {NTU_bat.item()}, mdot: {mdot_c.item()}, T_bat: {T_bat_noisy.item()}")
                            # Check the denominator specifically
                            print(f"Denom check: {1 - torch.exp(-NTU_bat).item()}")


                        mlp_input = torch.cat([
                        (T_bat_noisy/self.T_bat_scale).reshape(1,1),
                        (current/self.current_scale).reshape(1,1),
                        (omega/self.omega_scale).reshape(1,1),
                        (Q_heat/self.Q_heat_scale).reshape(1,1)
                        ], dim=1)
                        mlp_input.requires_grad_(True)

                        # Evaluate residual NN at current input
                        residual = self.residual_mlp(mlp_input)
                        # Calculate Jacobian wrt Temperature (1st element)
                        jacobian = torch.autograd.grad(
                        outputs=residual,
                        inputs=mlp_input,
                        grad_outputs=torch.ones_like(residual),
                        create_graph=True,
                        retain_graph=True
                        )[0]
                        jacobian_T = jacobian[:, 0]

                        jacobian_penalty = jacobian_T.pow(2).mean()

                        T_bat_pred = T_bat_noisy + dt*(T_bat_dot_model+ residual)

                        error = T_bat_next - T_bat_pred 
                        # Errors less than epsilon are disregarded
                        traj_loss +=  weight**i * torch.where(torch.abs(error)< epsilon, torch.zeros_like(error), error**2) 
                        jacobian_loss += self.lambda_jac*jacobian_penalty

                        residual_loss = residual_loss 
                        T_bat = T_bat_pred.squeeze()

                    T_bat = T_bat.detach().clone()
                loss += traj_loss + jacobian_loss
              
        
                loss.backward() 
                # Clipping gradient norm to avoid spiky networks
                nn.utils.clip_grad_norm_(
                    self.residual_mlp.parameters(),
                    max_norm= 0.1
                )

            
                self.residual_optimizer.step() # one optimization step to update parameters

                if epoch % 20 == 0:
                    print("Epoch {}: Training loss: {}".format(epoch, loss.item()))
            # Update residual NN
            self.l4c_residual.update(self.residual_mlp)
            

   
        if (self.current_iterate * self.dt) < self.T_warm_start:
            # Warm start (helps with initializing Simulink)
            if T_bat_0/self.T_bat_scale > T_bat_target: # Cooling regime
                omega_value, Q_heat_value = self.omega_scale*omega_max, self.Q_heat_scale*Q_heat_min
            else: # Heating regime
                omega_value, Q_heat_value = self.omega_scale*omega_max, self.Q_heat_scale*Q_heat_max
    

        self.xt_pred = np.array([T_bat_pred_nn])
        # Collect data on state, inputs and disturbance now, which in next iteration will be the last applied 
        self.T_bat_last = T_bat_0 
        self.omega_last = omega_value
        self.Q_heat_last = Q_heat_value
        self.current_last = disturbances[self.current_iterate].item()   

        T_bat_target = T_bat_target * self.T_bat_scale

        print(omega_value, Q_heat_value, self.xt_pred[0], T_bat_0, T_bat_target)
        print("total errors", self.total_errors)

        elapsed = 1000*(time.time() - start)
        print(elapsed, 'ms')
        print("--------------------------------")
        self.current_iterate += 1

        return omega_value, Q_heat_value, T_bat_pred_nn, T_bat_pred, pred_error_nn, pred_error, self.omega_scale*omega_norm_ss, self.Q_heat_scale*Q_heat_norm_ss, cost_to_go, elapsed, T_bat_dot_model, T_bat_dot_nn, dT_bat_euler, self.nn_on, slack_x