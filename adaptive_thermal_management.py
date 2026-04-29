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
import ctypes
import sys
import csv
import random as random
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error#mean_squared_error
from scipy.signal import savgol_filter


seed = sum([ord(char) for char in "DIAMONDS"])
np.random.seed(seed)
random.seed(seed)
CELSIUS_TO_KELVIN = 273.15
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=3):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim, bias=True), nn.Tanh()] # nn.ReLU rectified linear function (max(x,0))
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim, bias=True), nn.Tanh()]) # nn.Linear applies an affine transform. hidden_dim features and hidden_dim out features
        layers.append(nn.Linear(hidden_dim, output_dim, bias=True))
        self.net = nn.Sequential(*layers)

        #layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()] # nn.ReLU rectified linear function (max(x,0))
        #for _ in range(num_layers - 1):
        #    layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]) # nn.Linear applies an affine transform. hidden_dim features and hidden_dim out features
        #layers.append(nn.Linear(hidden_dim, output_dim))
        #self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
class BatteryLearnedDynamics:
    def __init__(self, residual_model): # remove gym_env for cascaded tank
        self.residual_model = residual_model
    def model(self, T_env): # remove gym_env for cascaded tank
        model = cs.types.SimpleNamespace()
        model.omega_scale = 100
        model.Q_heat_scale = 1000
        model.T_env = T_env
        T_bat = cs.MX.sym('T_bat')
        X = cs.vertcat(T_bat)

        omega_normalized = cs.MX.sym('omega_norm')
        omega = model.omega_scale * omega_normalized #cs.MX.sym('omega') # Power into heater/cooler given as efficiency*Pin
        Q_heat_normalized = cs.MX.sym('Q_heat_norm')
        Q_heat = model.Q_heat_scale * Q_heat_normalized #cs.MX.sym('Q_heat') # Pump control (rpm that is converted to kg/s)
        current = cs.MX.sym('current')
        nn_on = cs.MX.sym('nn_on') # Switch for when the NN is on

        P = cs.vertcat(current, nn_on)
                
        U = cs.vertcat(omega_normalized, Q_heat_normalized)
        
        nx = 1
        nu = 2

        # Parameters
        m_battery = 20*2.5*4
        c_battery = 795
        c_coolant = 3500
        density_coolant = 1050
        
        pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
        R_battery = 4*20*0.0128 # Battery resistance
        C_battery = 28*3600 # in coloumb
        hA_bat = 2500

        
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
        
        constraint = cs.types.SimpleNamespace()
        constraint.T_clin_min = -20 + CELSIUS_TO_KELVIN 
        constraint.T_clin_max = 100 + CELSIUS_TO_KELVIN
        constraint.T_clout_min = -20 + CELSIUS_TO_KELVIN
        constraint.T_clout_max = 100 + CELSIUS_TO_KELVIN
        constraint.expr = cs.vertcat(T_clin, T_clout)

        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        # Now for the actual calculations 
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))
        
        X_dot_nominal = cs.vertcat(T_bat_dot_model)
        T_bat_dot_function = cs.Function("f_model", [T_bat, current, omega_normalized, Q_heat_normalized], 
        [T_bat_dot_model], ["T_bat", "current", "omega_normalized", "Q_heat_normalized"], ["ode"])

        # MLP network
        #mlp_input = cs.vertcat(T_bat, current, U)
        #mlp_input = cs.vertcat(T_bat, current, U)
        mlp_input = cs.vertcat(T_bat, current, U) #T_bat #T_bat #Q_heat_normalized
        residual = self.residual_model(mlp_input.T).T 

        X_dot_residual = cs.vertcat(residual[0]) # x1 dot and x2 dot residual
        
        f_expl = X_dot_nominal + nn_on*X_dot_residual
        x_start = np.array([T_env]) # initial constraint (gets overwritten)

        # Power function (for cost)
        constant = 50 #18.1336
        Pump_power = (295.1748  * mdot_c**2  - 187.6638 * mdot_c+ constant)/(450)
        Heating_power = Q_heat/model.Q_heat_scale 
        cost = cs.vertcat(Pump_power, Heating_power, T_bat)
        cost_function = cs.Function("cost_function", [omega_normalized, Q_heat_normalized, T_bat], 
        [cost], ["omega_normalized", "Q_heat_normalized", "T_bat"], ["cost"])

        model.cost_y_expr_0 = cs.vertcat(Pump_power, Heating_power, T_bat)
        model.cost_y_expr = cs.vertcat(Pump_power, Heating_power, T_bat)
        model.cost_y_expr_e = cs.vertcat(T_bat)
        model.cost_function = cost_function
        # store to struct
        
        model.x = X 
        model.xdot = cs.MX.sym('xdot', nx)
        model.u = U
        model.z = cs.vertcat([])
        model.p = P
        model.f_expl = f_expl
        model.f_nominal = X_dot_nominal
        model.x_start = x_start
        model.constraints = cs.vertcat([]) # add constraints here or in mpc?
        model.name = "battery_economic"
        model.T_bat_dot_function = T_bat_dot_function
        return model, constraint 

 
class MPC:
    def __init__(self, model, constraint, N, t_horizon, external_shared_lib_dir, external_shared_lib_name):
        self.model = model
        self.constraint = constraint
        self.N = N
        self.t_horizon = t_horizon
        self.external_shared_lib_dir = external_shared_lib_dir
        self.external_shared_lib_name = external_shared_lib_name

    @property # a decorator
    def solver(self):
        return AcadosOcpSolver(self.ocp())
    
    def ocp(self):
        model = self.model
        constraint = self.constraint

        t_horizon = self.t_horizon
        N = self.N

        # Get model
        model_ac = self.acados_model(model=model, constraint=constraint)
        model_ac.con_h_expr = constraint.expr
        model_ac.con_h_expr_0 = constraint.expr

        # Dimensions
        #nx = 2
        nx = 1
        nu = 2
        # ny = nx + nu # Stage cost
        # ny_e = nx # Terminal cost considers only states
        nh = constraint.expr.shape[0]
        
        nsh = nh
        nsu = nu
        nsx = 1
        ns = nsu + nsx + nsh
        ny = 3
        ny_e = 1
        nsbx_e = 1
        ns_e = nsbx_e
        

        # Create ocp to formulate optim
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        # ocp.dims.ny = ny
        ocp.dims.nh = nh
        ocp.dims.ns = ns
        ocp.dims.nsbx = nsx
        ocp.dims.nsbu = nsu
        ocp.dims.ny = ny
        ocp.dims.ny_e = ny_e
        ocp.dims.nsbx_e = nsbx_e
        ocp.dims.ns_e = ns_e

        ocp.solver_options.tf = t_horizon

        # initialize cost function
        # ocp.cost.cost_type = 'LINEAR_LS'
        ocp.cost.cost_type_0 = 'NONLINEAR_LS'
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'


        x = ocp.model.x
        u = ocp.model.u
        

        x = ocp.model.x

        #ocp.cost.W = np.array([[1.]])
        constant = 50 #18.1336
        #Pump_power = (295.1748  * mdot_c**2  - 187.6638 * mdot_c+ constant)/(450)
        #Heating_power = Q_heat/model.Q_heat_scale 
        #cost = cs.vertcat(Pump_power, Heating_power, T_bat)
        # Trivial PyTorch index 0
        density_coolant = 1050
        pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
        l4c_y_expr_row = l4c.L4CasADi(lambda u0: (295.1748  * (density_coolant*pump_displacement*u0)**2  - 187.6638 * (density_coolant*pump_displacement*u0)+ constant)/(450), name='y_expr_1')
        ocp.model.cost_y_expr = cs.vertcat(l4c_y_expr_row(u[0]), u[1], x)
        ocp.model.cost_y_expr_0 = cs.vertcat(l4c_y_expr_row(u[0]), u[1], x)
        ocp.model.cost_y_expr_e = x
        
        #l4c_y_expr_row_1 = l4c.L4CasADi(lambda u1: u1, name='y_expr_2')
        #l4c_y_expr_row_1 = l4c.L4CasADi(lambda u1: (295.1748  * (density_coolant*pump_displacement*u1)**2  - 187.6638 * (density_coolant*pump_displacement*u1)+ constant)/(450), name='y_expr_3')
        
        #l4c_y_expr = l4c.L4CasADi(model_ac.cost_function, name='y_expr')
        #print("type l4cexpr")
        #print(type(l4c_y_expr))
        #print(l4c_y_expr.model)
        #print(type(l4c_y_expr.model))
        #inputs = cs.vertcat(u[0], u[1], x)
        #ocp.model.cost_y_expr = l4c_y_expr(inputs.T)     #print("type l4c_expr_called(x)")
        #print(type(l4c_y_expr(x)))
        #ocp.model.cost_y_expr_e = x
        # # state 
        # ocp.cost.Vx = np.zeros((ny, nx)) # i feel like this should be nx,nx ||Vx x||^2 
        # for i in range(nx):
        #     ocp.cost.Vx[i,i] = 1 # so set the matrix coeff corr to u to 0. maybe the dim of Vx is set to be the same as Vu
        # ocp.cost.Vu = np.zeros((ny, nu))
        # for i in range(nu):
        #     ocp.cost.Vu[i + nx, i] = 1
        # ocp.cost.Vz = np.array([[]]) # don't know what the V_z z, what the variable z should be
        # ocp.cost.Vx_e = np.eye(nx)
        # #l4c_y_expr = None

        # Define weight parameters
        # Q = np.diag([10, 0.1])
        # R = np.diag([1,1])
        Q = np.array([1.0, 1.0, 0.01])
        Q = np.diag(Q)
        # ocp.cost.W = scipy.linalg.block_diag(Q,R)
        # ocp.cost.W_e = Q 
        ocp.cost.W_0 = Q
        ocp.cost.W = Q
        ocp.cost.W_e = np.diag(np.array([0.01]))

        ocp.cost.yref_0 = np.zeros((ny, ))
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # Initial state
        ocp.constraints.x0 = model.x_start

        # Set constraints
        omega_max = (4000*2*np.pi/60)/model.omega_scale
        
        omega_min = (150*2*np.pi/60)/model.omega_scale
        #omega_min = (2000*2*np.pi/60)/model.omega_scale
        
        print("omega max", omega_max)
    
        Q_heat_max = 4000/model.Q_heat_scale
        print("Q_heat max", Q_heat_max)
        Q_heat_min = -4000/model.Q_heat_scale
        Tb_max = 50 + CELSIUS_TO_KELVIN
        Tb_min = -20 + CELSIUS_TO_KELVIN
        SOC_max = 1
        SOC_min = 0

        # ocp.constraints.ubx = np.array([Tb_max])
        # ocp.constraints.lbx = np.array([Tb_min])
        
        # Nonlinear constraints (for coolant)

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

        # Note: We do not have terminal nonlinear constraint
        # This is because the nonlinear constraint includes inputs
        # For terminal node we do not have inputs so we cannot use it 

        # Add terminal constraints
        T_steady_low = 19 + CELSIUS_TO_KELVIN
        T_steady_high = 22 + CELSIUS_TO_KELVIN

        # Constraints on inputs
        ocp.constraints.idxbu = np.array([0,1])
        ocp.constraints.lbu = np.array([omega_min, Q_heat_min])
        ocp.constraints.ubu = np.array([omega_max, Q_heat_max])
        # Constraints on state 
        ocp.constraints.idxbx = np.array([0]) # at what indices to have constraints
        ocp.constraints.lbx = np.array([T_steady_low]) 
        ocp.constraints.ubx = np.array([T_steady_high])
        # Add terminal constraints 
        ocp.constraints.idxbx_e = np.array([0]) # at what indices to have constraints
        ocp.constraints.lbx_e = np.array([T_steady_low]) 
        ocp.constraints.ubx_e = np.array([T_steady_high])
        


        # Add slack to states
        slack_allowable_low = Tb_min - T_steady_low  # How far below steady-state is allowed 
        slack_allowable_high = Tb_max - T_steady_high  # How far above steady-state is allowed 
        
        #ocp.constraints.lsbx = np.array([slack_allowable_low])   # Slack lower bound
        #ocp.constraints.usbx = np.array([slack_allowable_high])  # Slack upper bound

        # Add slack to terminal constraints
        #ocp.constraints.lbsx_e = np.array([slack_allowable_low]) 
        #ocp.constraints.ubsx_e = np.array([slack_allowable_high])
        

        
        # Slack constraint indices
        ocp.constraints.idxsbx = np.array(range(nsx)) # Slack on state for nodes 1 --- (N - 1)
        ocp.constraints.idxsbx_e = np.array(range(nsx)) # Slack on terminal shooting for state
        # Note, no "idxbx_0" so have not slack on initial state. But makes sense since we give solver what x0 must be
        ocp.constraints.idxsh_0 = np.array(range(nsh)) # Slack on initial shooting for nonlinear constraint
        ocp.constraints.idxsh = np.array(range(nsh)) # Slack on nonlinear constraint

        ocp.constraints.idxsbu = np.array(range(nsu)) # Slack on u

        
        # Slack cost
        ocp.cost.zl = 10000 * np.zeros((ns,))
        ocp.cost.zu = 10000 * np.zeros((ns,))
        ocp.cost.Zl = 10000 * np.ones((ns,))
        ocp.cost.Zu = 10000 * np.ones((ns,))
        #ocp.cost.Zl[1] = 1
        ocp.cost.Zl[2] = 100
        #ocp.cost.Zu[1] = 1
        ocp.cost.Zu[2] = 100
        
        ocp.cost.zl_0 = 10000 * np.zeros((nsh+nsu,))
        ocp.cost.zu_0 = 10000 * np.zeros((nsh+nsu,))
        ocp.cost.Zl_0 = 10000* np.ones((nsh+nsu,))
        ocp.cost.Zu_0 = 10000* np.ones((nsh+nsu,))

        ocp.cost.zl_e = 1000 * np.zeros((nsx,))
        ocp.cost.zu_e = 1000 * np.zeros((nsx,))
        ocp.cost.Zl_e = 1000* np.ones((nsx,))
        ocp.cost.Zu_e = 1000 * np.ones((nsx,))

        # Solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.sim_method_num_stages = 1
        ocp.solver_options.regularize_method = 'MIRROR' # Didn't immediately help
        ocp.solver_options.reg_epsilon = 1e-3

        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name + ' -l' + l4c_y_expr_row.name
        ocp.solver_options.sim_method_num_stages = 1
        
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
        model_ac.cost_y_expr = model.cost_y_expr
        model_ac.cost_y_expr_0 = model.cost_y_expr_0
        model_ac.cost_y_expr_e = model.cost_y_expr_e
        model_ac.cost_function = model.cost_function 
        return model_ac



class Controller:
    def setup(self, T_bat_target, T_env):
        
        # Residual MLP: Lightweight
        residual_mlp = MLP(input_dim = 4, output_dim=1, hidden_dim=16, num_layers=2) # the network
        
        #residual_mlp = MLP(input_dim = 2 + 2, output_dim=1, hidden_dim=16, num_layers=1) # the network
        for param in residual_mlp.parameters():
            param.requires_grad = False
        residual_mlp.load_state_dict(torch.load("heating_pretrain_full_network.pth", weights_only=True))
        self.residual_mlp = residual_mlp
        self.residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=1e-2, weight_decay=0.1) # lr = learning rate, the optimizer
        #self.residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=1e-3) # lr = learning rate, the optimizer
        #print(residual_mlp[0])
        self.residual_criterion = nn.MSELoss()

        l4c_residual = l4c.L4CasADi(self.residual_mlp, name="battery_economic", mutable=True)
        self.l4c_residual = l4c_residual
        print(l4c_residual)
        # MPC Setup 
        self.N = 200 # 80
        self.t_horizon = self.N*5
        learned_model = BatteryLearnedDynamics(l4c_residual)
      
        casadi_model, constraint = learned_model.model(T_env=T_env)
        self.T_bat_dot_function = casadi_model.T_bat_dot_function

        # MIGHT CAUSE ISSUES SINCE DIFFERENT THAN YU MEI, here model=casadi_model
        self.solver = MPC(model=casadi_model, constraint=constraint, N=self.N, t_horizon = self.t_horizon,
                    external_shared_lib_dir=l4c_residual.shared_lib_dir,
                    external_shared_lib_name=l4c_residual.name).solver # Returns the solver object from MPC

        # SOC_ref = None # This is not actually tracked
        self.dt = self.t_horizon/self.N
        self.obs_buffer = []
        self.batch_size = 40 #30
        self.T_update = self.batch_size*self.dt
        self.T_warm_start = 30
        self.current_iterate = 0
        self.xt_pred = np.array([CELSIUS_TO_KELVIN, 1])

        self.x_last = np.array([0,0])
        self.omega_last = 0
        self.Q_heat_last = 0
        self.current_last = 0
        self.T_bat_last = 0
        
        self.data = []
        self.total_errors = 0
        self.total_cost = 0
        self.omega_scale = casadi_model.omega_scale
        self.Q_heat_scale = casadi_model.Q_heat_scale
        self.T_env = T_env
        self.nn_on = 0
        # Reading disturbance info
        df = pd.read_csv("current_intp1.csv", names=["current"])
        arr = df.to_numpy(dtype=np.float32)
        self.disturbance_values = arr

    def get_input(self, T_bat_target, T_bat_0, SOC_0, current_0, dT_bat, dSOC):
       
        #print("--------------------------------")
        #print("ADAPTIVE ACADOS MPC BATTERY MODEL WITH TARGET TRACKING AND COST-TO-GO")
        start = time.time()

        if self.current_iterate == 0:
            self.xt_pred = ([T_bat_0])       
            # Warm start
            omega_warm_start = (2000*2*np.pi/60)/self.omega_scale
            if T_bat_0 <= 20.5:
                Q_heat_warm_start = 2000/self.Q_heat_scale
            else:
                Q_heat_warm_start = -2000/self.Q_heat_scale
            input_warm_start = np.array([omega_warm_start, Q_heat_warm_start])
            self.solver.set(0, 'u', input_warm_start) # Weird but this is how acados does warm starts. Does not "force" inputs to be this
            state_warm_start = np.array([T_bat_0])
            self.solver.set(0, 'x', state_warm_start)
        disturbances = self.disturbance_values
        T_env = self.T_env
        # Set reference for each step in MPC horizon
        #omega_norm_ss, Q_heat_norm_ss = self.get_steady_state(T_bat_target=T_bat_target)
        #Q_e, cost_to_go = self.get_cost_to_go(T_bat_target, T_env, omega_norm_ss, Q_heat_norm_ss)
        #print("omega_ss", omega_norm_ss, "Q_heat_ss", Q_heat_norm_ss)
        #print("Q_e", Q_e)
        if self.nn_on:
            pass
            #print("NN on from hereafter")
        for k in range(self.N):
            #if k == 0: # only store first reference
                #Tb_ref_history.append(Tb_ref)
                # SOC_ref_history.append(SOC_ref)
            # Set terminal reference
            # y_ref_k = np.array([Tb_ref, SOC_ref, 0])
            y_ref_k = np.array([0,0, T_bat_target])  
         
            self.solver.set(k, "yref", y_ref_k)
            # Potential for error
            # T_env must be in Kelvin
            #param_values = np.array([disturbances[int(self.dt*(self.current_iterate + k))].item(), self.nn_on]) 
            param_values = np.array([0, self.nn_on]) 

            self.solver.set(k, "p", param_values)

        # Set terminal reference
        # y_ref_terminal = np.array([Tb_ref, SOC_ref])
        y_ref_terminal = np.array([T_bat_target]) # Terminal cost only on states so 2x1 instead of 4x1 above
        #print(y_ref_terminal)
        self.solver.set(self.N, "yref", y_ref_terminal)
        param_values = np.array([0, self.nn_on])
        #param_values = np.array([disturbances[int(self.dt*(self.current_iterate + self.N))].item(), self.nn_on])

        self.solver.set(self.N, "p", param_values)

        #self.solver.cost_set(self.N, 'W', Q_e)

        xt = np.array([T_bat_0])
        #self.solver.set(0, "x", xt)
        

        # Apply current state as constraint
        self.solver.set(0, "lbx", xt)
        self.solver.set(0, "ubx", xt)
        #s_Q_heat_norm = 0/self.Q_heat_scale
        #s_omega_norm = 0/self.omega_scale
        # Constraint tightening
        #if (T_bat_0 < (18.5 + CELSIUS_TO_KELVIN)) or (T_bat_0 > (22.5 + CELSIUS_TO_KELVIN)):
            #s_Q_heat_norm = 200/self.Q_heat_scale
            #s_omega_norm = 20/self.omega_scale
        #elif ((18.5 + CELSIUS_TO_KELVIN) <= T_bat_0 < (19.5 + CELSIUS_TO_KELVIN)) or (22.5 + CELSIUS_TO_KELVIN >= T_bat_0 > (21.5 + CELSIUS_TO_KELVIN)):
            #s_Q_heat_norm = 20/self.Q_heat_scale
            #s_omega_norm = 2/self.omega_scale
        # Comment out for excitation
        #s_Q_heat_norm = 0/self.Q_heat_scale
        #s_omega_norm = 0/self.omega_scale
        #s_norm = np.array([s_omega_norm, s_Q_heat_norm])
        # Set constraints
        omega_min, omega_max = (150*2*np.pi/60)/self.omega_scale, (4000*2*np.pi/60)/self.omega_scale
            
        Q_heat_min, Q_heat_max = -4000/self.Q_heat_scale, 4000/self.Q_heat_scale
       
        #u_min_tightened = np.array([omega_min, Q_heat_min]) + s_norm
        #u_max_tightened = np.array([omega_max, Q_heat_max]) - s_norm
        #self.solver.constraints_set(0, 'lbu', u_min_tightened)
        #self.solver.constraints_set(0, 'ubu', u_max_tightened)
        # Solve mpc and apply control
        self.solver.solve()
        self.total_cost += self.solver.get_cost()
        # ut = solver.get(0, "u").item()
        ut = self.solver.get(0, "u")
        #print("SOLVER solution", ut)
        shooting_node = 1
        slack_lower = self.solver.get(shooting_node, "sl")
        slack_upper = self.solver.get(shooting_node, "su")
        slx = slack_upper[2]
        print("Slacks for shooting node {}".format(shooting_node))
        print("slack lower", slack_lower, "slack upper", slack_upper)
        
        #print("slack lower", slack_lower, "slack upper", slack_upper)
        
        # Saturation
        #omega_value = min(ut[0].item(), omega_max - s_omega_norm)
        #Q_heat_value = min(ut[1].item(), Q_heat_max - s_Q_heat_norm)
        #omega_value = max(omega_value, omega_min + s_omega_norm)
        #Q_heat_value = max(Q_heat_value, Q_heat_min + s_Q_heat_norm)
        omega_value = min(ut[0].item(), omega_max)
        Q_heat_value = min(ut[1].item(), Q_heat_max)
        omega_value = max(omega_value, omega_min)
        Q_heat_value = max(Q_heat_value, Q_heat_min)
        omega_value = self.omega_scale * omega_value
        Q_heat_value = self.Q_heat_scale * Q_heat_value
        #print("inputs normalized", ut)
        status = self.solver.get_status()
        if status != 0:
            #print("ERROR", status, "at iterate", self.current_iterate)
            self.total_errors += 1
        # Want to compare prediction at time step 0 with value at time step 1
        # So delay update of xt pred
       
        pred_error = 0
        pred_error_nn = 0
        T_bat_pred = 0
        T_bat_pred_nn = 0
        # Collect training data for NN

        
        if self.current_iterate  > 0:

            # input and disturbance values (not symbolics because know what happened)
            # Not sure if should take current values or last, but figure that at this moment the change is happening because of the last values
            omega = self.omega_last
            Q_heat = self.Q_heat_last 
            #current = self.current_last
            #T_bat = self.T_bat_last
            
            
            current = self.current_last
            T_bat = self.T_bat_last
            T_bat_k_minus_1 = self.T_bat_last
            T_bat_k = T_bat_0

            # Approximate derivative
            dT_bat_euler = (T_bat_k - T_bat_k_minus_1)/self.dt
            #print("derivative read from model", dT_bat, "derivative euler step", dT_bat_euler)


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

            T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(self.T_env - self.T_bat_last))
            #T_bat = 100*(0.5 - random.random()
            self.data.append((T_bat, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale))
            #self.data.append((T_bat, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale))
            
            self.residual_mlp.eval()

            #current = disturbances[int(self.dt*(self.current_iterate))].item()
            #residual_data = torch.tensor([T_bat_k_minus_1, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale], dtype=torch.float32)
            #residual_data = torch.tensor([Q_heat/self.Q_heat_scale], dtype=torch.float32)
            residual_data = torch.tensor([T_bat, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale], dtype=torch.float32)
            
            residual = self.residual_mlp(residual_data).numpy().item()
            T_bat_dot_nn = T_bat_dot_model + residual
            

            #T_bat_pred_nn =  T_bat_k_minus_1 + self.dt * (K1 + self.nn_on*residual)
           
            pred_error_nn = 0 #T_bat_pred_nn - T_bat_k
            pred_error = T_bat_k - T_bat_pred  
            pred_error_dt = pred_error/self.dt
            #self.obs_buffer.append((pred_error_dt))
            dynamics_residual = dT_bat_euler - T_bat_dot_model
            self.obs_buffer.append((dynamics_residual))
            self.residual_mlp.train()

        else:
            #self.obs_buffer.append((0, 0))
            self.obs_buffer.append((0))
            self.data.append((0,0,0,0))
            #self.data = [(0,0,0,0)]
            T_bat_dot_model = dT_bat
            dT_bat_euler = dT_bat
            T_bat_dot_nn = dT_bat

        # update every T_update = 60 
        # T_warm_start = 30
        if (self.dt*self.current_iterate) > self.T_warm_start and ((self.dt*self.current_iterate) % self.T_update) == 0 and len(self.obs_buffer) >= self.batch_size:
            self.nn_on = 1
            data = np.array(self.data[-self.batch_size:])
            #data = data.reshape((-1,1))
            obs = np.array(self.obs_buffer[-self.batch_size:])
            print("data")
            print(data)
            print(data.shape)
            print("obs")
            print(obs)
            print(obs.shape)
            X_train, X_test, y_train, y_test = train_test_split(data, obs, test_size=0.25)

            # data[:, :3] h1, h2 and u
            #X_batch = torch.tensor(data[:, :], dtype=torch.float32)
            X_batch = torch.tensor(X_train, dtype=torch.float32)
            print(y_train)
            print(y_train.shape)
            # h1_dot, h2_dot
            #y_true = y_train[:,0]
            #y_nominal = y_train[:,1]
            y_target = y_train
            polyorder = 3
            window_length = 10
            y_target = savgol_filter(y_target, window_length=window_length, polyorder=polyorder)
            y_train = y_train.reshape((-1,1))
            
            #y_true = y_true.reshape((-1,1))
        
            

            #y_nominal = y_nominal.reshape((-1,1))
            #y_target = obs[-self.batch_size:]
            #y_target = y_target.reshape((-1,1))
            
            y_target = torch.tensor(y_target, dtype=torch.float32)
            print(X_batch, y_target)
            print(X_batch)
            print(y_target)


            
            for p in self.residual_mlp.parameters(): p.requires_grad = True
            for _ in range(200): #200 before
                self.residual_optimizer.zero_grad() # optimizer object
                prediction = self.residual_mlp(X_batch) # gives data to network to make a prediction
                loss = self.residual_criterion(prediction, y_target)
                #l1_norm = sum(torch.linalg.norm(p, 1) for p in self.residual_mlp.parameters())
                #l2_norm = sum(p.pow(2).sum() for p in self.residual_mlp.parameters())
                #regularization = 0.1
                #loss += regularization * l1_norm
                loss.backward() # calculates gradient
                
                self.residual_optimizer.step() # one optimization step to update parameters
            for p in self.residual_mlp.parameters(): 
                p.requires_grad = False
                print(p)
            #print("UPDATE MODEL AT", self.dt*self.current_iterate)
            
            self.l4c_residual.update(self.residual_mlp)

            test_data = torch.tensor(X_test, dtype=torch.float32)
            #y_true_test = y_test[:,0]
            #y_true_test = y_true_test.reshape((-1,1))
        
            

            #y_nominal_test = y_test[:,1]
            #y_nominal_test = y_nominal_test.reshape((-1,1))
            #y_target = obs[-self.batch_size:]
            #y_target = y_target.reshape((-1,1))
            
            y_target_test = torch.tensor(y_test, dtype=torch.float32)
            self.residual_mlp.eval()
            with torch.no_grad():
                outputs = self.residual_mlp(test_data)
                target_predicted = np.array(outputs.squeeze().tolist())
            mse = root_mean_squared_error(y_target_test, target_predicted)
            len_test = len(y_test)
            null_prediction = np.zeros((len_test,))
            print("MSE", mse)
            mse_null_hypothesis = root_mean_squared_error(y_target_test, null_prediction)
            print("MSE null hypothesis", mse_null_hypothesis)
            self.residual_mlp.train()

            if mse >= mse_null_hypothesis:
                self.nn_on = 0
        # Applying random noise for better excitation
        #s_omega = s_omega_norm*2*(random.random() - 0.5)
        #s_Q_heat = s_Q_heat_norm*2*(random.random() - 0.5)
        #print("random input noise", self.omega_scale * s_omega, self.Q_heat_scale * s_Q_heat)
        omega_value = omega_value #+ self.omega_scale * s_omega
        Q_heat_value = Q_heat_value #+ self.Q_heat_scale * s_Q_heat
        if (self.current_iterate * self.dt) < self.T_warm_start or (status != 0):
            #print("WARM STARTING")
            if T_bat_0 > T_bat_target: # Cooling regime
                omega_value, Q_heat_value = self.omega_scale*omega_max, self.Q_heat_scale*Q_heat_min
            else: # Heating regime
                omega_value, Q_heat_value = self.omega_scale*omega_max, self.Q_heat_scale*Q_heat_max
    

        self.xt_pred = np.array([T_bat_pred_nn])
        self.T_bat_last = T_bat_0 # Changed from xt[0], should be the same but for sanity
        self.omega_last = omega_value
        self.Q_heat_last = Q_heat_value
        #self.current_last = disturbances[int(self.dt*(self.current_iterate))].item()
        self.current_last = 0 # disturbances[int(self.dt*(self.current_iterate))].item()

        #print(omega_value, Q_heat_value, self.xt_pred[0], T_bat_0, T_bat_target)
        #print("time", self.dt*self.current_iterate)
        #print("cumulative cost", self.total_cost)
        print("total errors", self.total_errors)

        elapsed = 1000*(time.time() - start)
        print(elapsed, 'ms')
        #print("--------------------------------")
        self.current_iterate += 1

        return omega_value, Q_heat_value, T_bat_pred_nn, T_bat_pred, pred_error_nn, pred_error, 0, 0, 0, elapsed, T_bat_dot_model, T_bat_dot_nn, dT_bat_euler, self.nn_on