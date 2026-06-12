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
from sklearn.metrics import root_mean_squared_error
from scipy.signal import savgol_filter
from scipy import signal


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

### Model of battery dynamics 
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
        mlp_input = cs.vertcat(T_bat_normalized, current_normalized, omega_normalized, Q_heat_normalized) 
        residual = self.residual_model(mlp_input.T).T 

        X_dot_residual = cs.vertcat(residual[0]) # residual
        
        # Adding residual neural network to dynamics. Switch turned on after first training of NN
        f_expl = (X_dot_nominal + nn_on*X_dot_residual)/model.T_bat_scale
        x_start = np.array([T_env/model.T_bat_scale]) # initial constraint (gets overwritten)

        ## Power functions (for cost)
        Pump_power = (-0.21574 + 10.0501*omega_normalized**2 + 6.84987*omega_normalized**3)/1000 # Casadi pump fit 3rd degree, / 1000 to obtain in Kilowatts
        Heating_power = Q_heat/model.Q_heat_scale # Q_heat / Q_heat_scale = [W / 1000] = [kW]
        cost = cs.vertcat(Pump_power, Heating_power, T_bat)
        cost_function = cs.Function("cost_function", [omega_normalized, Q_heat_normalized, T_bat_normalized], 
        [cost], ["omega_normalized", "Q_heat_normalized", "T_bat"], ["cost"])
        # y_expr will be used in cost function. Vector of [Pump power; Heating power; T_bat]
        model.cost_y_expr_0 = cs.vertcat(Pump_power, Heating_power, T_bat_normalized)
        model.cost_y_expr = cs.vertcat(Pump_power, Heating_power, T_bat_normalized)
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
        model.constraints = cs.vertcat([])
        model.name = "battery_economic"
        model.T_bat_dot_function = T_bat_dot_function
        return model, constraint 

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
        nh = constraint.expr.shape[0]
        
        nsh = nh
        nsu = nu
        nsx = 1
        ns = nsu + nsx + nsh
        ny = 3 # Stage cost variable dimension
        ny_e = 1 # Terminal cost considers only states
        nsbx_e = 1
        ns_e = nsbx_e
        

        # Create ocp to formulate optim
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.nh = nh
        ocp.dims.ns = ns
        ocp.dims.nsbx = nsx
        ocp.dims.nsbu = nsu
        ocp.dims.ny = ny
        ocp.dims.ny_e = ny_e
        ocp.dims.nsbx_e = nsbx_e
        ocp.dims.ns_e = ns_e

        ocp.solver_options.tf = t_horizon
        x = ocp.model.x
        u = ocp.model.u
        
        # initialize cost function
        # NONLINEAR_LS to obtain an economic formulation
        ocp.cost.cost_type_0 = 'NONLINEAR_LS'
        ocp.cost.cost_type = 'NONLINEAR_LS'
        ocp.cost.cost_type_e = 'NONLINEAR_LS'
        # See problem_formulation_ocp in acados docs (available at https://github.com/acados/acados/tree/main/docs/problem_formulation)
        # But summary
        # Stage cost = ||y_expr - y_ref||^2_W, y_expr is a casadi expression.
        # Equivalent for final cost and initial cost 
        # Here y_expr = [Pump power; Heater Power; State]

        # L4CasADi needs to autogenerate code, such as jacobians
        # For this, the y_expr needs to be formulated as an anonymous function 
        # Note that only the pump power is a "function" of omega, i.e. first input u[0], 
        # Q_heat_power is just the 2nd input (u[1]) and T_bat is just the state x
        l4c_y_expr_row_1 = l4c.L4CasADi(lambda u0: (-0.21574 + 10.0501*u0**2 + 6.84987*u0**3)/(1000), name='y_expr_1')
        
        ocp.model.cost_y_expr = cs.vertcat(l4c_y_expr_row_1(u[0]), u[1], x)
        ocp.model.cost_y_expr_0 = cs.vertcat(l4c_y_expr_row_1(u[0]), u[1], x)
        ocp.model.cost_y_expr_e = x
     
        # Define weight matrices
        Q = np.array([15, 15, 0]) # note: no weighing on state
        Q = np.diag(Q)

        ocp.cost.W_0 = Q
        ocp.cost.W = Q
        ocp.cost.W_e = np.diag(np.array([0.0]))  # Terminal cost only considers state, set this to 0 for no weighing
        ocp.cost.yref_0 = np.zeros((ny, ))
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))
        # Initial state
        ocp.constraints.x0 = model.x_start

        # Set constraints
        omega_max = (4000*2*np.pi/60)/model.omega_scale
        
        omega_min = (150*2*np.pi/60)/model.omega_scale

        Q_heat_max = 4000/model.Q_heat_scale
        Q_heat_min = -4000/model.Q_heat_scale
        
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
        T_steady_low = (12 + CELSIUS_TO_KELVIN)/model.T_bat_scale
        T_steady_high = (28 + CELSIUS_TO_KELVIN)/model.T_bat_scale

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
        


        # Add slack to states (to have a target region formulation)
        # Slack constraint indices
        ocp.constraints.idxsbx = np.array(range(nsx)) # Slack on state for nodes 1 --- (N - 1)
        ocp.constraints.idxsbx_e = np.array(range(nsx)) # Slack on terminal shooting for state
        # Note, no "idxbx_0" so have not slack on initial state. But makes sense since we give solver what x0 must be

        ocp.constraints.idxsh_0 = np.array(range(nsh)) # Slack on initial shooting for nonlinear constraint
        ocp.constraints.idxsh = np.array(range(nsh)) # Slack on nonlinear constraint
        
        # Slack cost
        ocp.cost.zl = 1000 * np.zeros((nsx+nsh,))
        ocp.cost.zu = 1000 * np.zeros((nsx+nsh,))
        ocp.cost.Zl = 1000 * np.ones((nsx+nsh,))
        ocp.cost.Zu = 1000 * np.ones((nsx+nsh,))
       
        ocp.cost.zl[0] = 1000
        ocp.cost.zu[0] = 1000
        
        ocp.cost.Zl[0] = 100
        ocp.cost.Zu[0] = 100

        ocp.cost.zl_0 = 1000 * np.zeros((nsh,))
        ocp.cost.zu_0 = 1000* np.zeros((nsh,))
        ocp.cost.Zl_0 = 1000* np.ones((nsh,))
        ocp.cost.Zu_0 = 1000* np.ones((nsh,))

        ocp.cost.zl_e = 1000 * np.ones((nsx,))
        ocp.cost.zu_e = 1000 * np.ones((nsx,))
        ocp.cost.Zl_e = 100* np.ones((nsx,))
        ocp.cost.Zu_e = 100* np.ones((nsx,))

        # Solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.sim_method_num_stages = 1
        # Regularization of cost function, maybe not strictly necessary
        ocp.solver_options.regularize_method = 'MIRROR' 
        ocp.solver_options.reg_epsilon = 1e-3
        # Libraries (folders) for auto-generated L4CasADi code, e.g. jacobians
        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name + ' -l' + l4c_y_expr_row_1.name

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


### Controller including setup, collect_data and get_input
class Controller:
    def setup(self, T_bat_target, T_env):
        # MPC Setup 
        self.N = 200 
        self.t_horizon = self.N*5      
        self.dt = self.t_horizon/self.N
        # NN architecture and hyperparameter
        learning_rate = 1e-3
        input_dim = 4
        output_dim = 1
        hidden_dim = 32
        num_layers = 2 # number of hidden layers
        weight_decay = 0.01

        self.lambda_jac = 0.1 # Weight of jacobian loss
        self.epsilon = 0.2 # Noise added in training 
        self.noise_std = 0.001
        self.confidence = 0.001 # Confidence in neural network
        self.T_rollout = 20 # Rollout length (how often re-initialize true temperature in trajectory simulation)
        # Amount of data to save before training NN
        self.batch_size = 200 
        self.T_update = self.batch_size*self.dt # How often NN is updated
        # Residual MLP
        residual_mlp = MLP(input_dim = input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers) # the network
        # Freeze parameters        
        for param in residual_mlp.parameters():
            param.requires_grad = False
        residual_mlp.load_state_dict(torch.load("Offline_Training/Pretrained_Networks/economic_pretrain_FINAL.pth", weights_only=True))
        self.residual_mlp = residual_mlp
        self.residual_optimizer = torch.optim.AdamW(residual_mlp.parameters(), lr=learning_rate, weight_decay=weight_decay) 
        self.residual_criterion = nn.MSELoss()

        l4c_residual = l4c.L4CasADi(self.residual_mlp, name="battery_economic", mutable=True)
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
        self.xt_pred = np.array([CELSIUS_TO_KELVIN, 1])

        self.x_last = np.array([0,0])
        self.omega_last = 0
        self.Q_heat_last = 0
        self.current_last = 0
        self.T_bat_last = 0
        self.dynamics = []
        self.data = []
        self.total_errors = 0
        self.total_cost = 0
        self.omega_scale = casadi_model.omega_scale
        self.Q_heat_scale = casadi_model.Q_heat_scale
        self.T_bat_scale = casadi_model.T_bat_scale
        self.current_scale = casadi_model.current_scale
        self.T_env = T_env
        self.nn_on = 0

        # Reading disturbance info
        df = pd.read_csv("current_rms_5s.csv", names=["current"])
        arr = df.to_numpy(dtype=np.float32)
        self.disturbance_values = arr

    def get_input(self, T_bat_target, T_bat_0, SOC_0, current_0, dT_bat, dSOC):
        # MPC step that returns input (and other values for plotting)       
        print("--------------------------------")
        print("ADAPTIVE ECONOMIC TRAJECTORY LOSS MPC BATTERY MODEL")
        start = time.time()

        if self.current_iterate == 0:
            # Warm start solver for iterate = 0
            self.xt_pred = ([T_bat_0])       
            # Warm start
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
        # Set reference and current (disturbance) for each step in MPC horizon

        for k in range(self.N):
            # Power_target = 0 to minimize power consumption

            y_ref_k = np.array([0,0, T_bat_target/self.T_bat_scale])  
         
            self.solver.set(k, "yref", y_ref_k)

            param_values = np.array([disturbances[self.current_iterate + k].item()/self.current_scale, self.nn_on])

            self.solver.set(k, "p", param_values)

        # Set terminal reference
        y_ref_terminal = np.array([0]) # Terminal cost only on states
        self.solver.set(self.N, "yref", y_ref_terminal)
        # Set current on last time step
        param_values = np.array([disturbances[self.current_iterate + k].item()/self.current_scale, self.nn_on])
        self.solver.set(self.N, "p", param_values)


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
        print("SOLVER solution", ut)
        
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
        # Collect training data for NN
        if self.current_iterate  >= 0:
            
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
            residual_data = torch.tensor([T_bat/self.T_bat_scale, current/self.current_scale, omega/self.omega_scale, Q_heat/self.Q_heat_scale], dtype=torch.float32)
            
            residual = self.residual_mlp(residual_data).numpy().item()
            T_bat_dot_nn = T_bat_dot_model + residual
            K1 = cs.DM.full(self.T_bat_dot_function(T_bat/self.T_bat_scale, current/self.current_scale, omega/self.omega_scale, Q_heat/self.Q_heat_scale)).item()
            
            T_bat_pred = T_bat + self.dt*(K1)
            T_bat_pred_nn = T_bat + self.dt*T_bat_dot_nn
            pred_error_nn = T_bat_0 - T_bat_pred_nn 
            pred_error = T_bat_0 - T_bat_pred   
            pred_error_dt = pred_error/self.dt
            self.residual_mlp.train()

        # NN training
        # Occurs after a batch of batch size has been collected, and every T_update, and only after warm start has been finished
        if (self.dt*self.current_iterate) > self.T_warm_start and ((self.dt*self.current_iterate) % self.T_update) == 0 and len(self.data) >= self.batch_size:
            self.nn_on = 1 # NN on switch
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
            # Uncomment if want to use calculated confidence, and change it automatically. 
            #np.sqrt(np.power(dynamics_savgol-dynamics_model,2).mean())
            with torch.no_grad():
                self.residual_mlp.tau.copy_(torch.tensor([confidence]))              
            print("confidence", confidence)
            
            temperatures_tensor = torch.tensor(temperatures_data, dtype=torch.float32)
            current_tensor = torch.tensor(current_filtered,  dtype=torch.float32)
            omega_scaled_tensor = torch.tensor(omega_scaled_data, dtype=torch.float32)
            Q_heat_scaled_tensor = torch.tensor(Q_heat_scaled_data, dtype=torch.float32)

            T_rollout = self.T_rollout
            lambda_jac = self.lambda_jac
            epsilon = self.epsilon
            noise_std = self.noise_std

            # Epoch loop
            for epoch in range(200):
                self.residual_optimizer.zero_grad() 
                loss = 0
                traj_loss = 0
                jacobian_loss = 0
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

                        jacobian_loss += lambda_jac*jacobian_penalty

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
            


        if (self.current_iterate * self.dt) < self.T_warm_start or (status != 0):
            # Warm start (helps with initializing Simulink)
            if T_bat_0 > T_bat_target: # Cooling regime
                omega_value, Q_heat_value = self.omega_scale*omega_max, self.Q_heat_scale*Q_heat_min
            else: # Heating regime
                omega_value, Q_heat_value = self.omega_scale*omega_max, self.Q_heat_scale*Q_heat_max
    

        self.xt_pred = np.array([T_bat_pred_nn])
        self.T_bat_last = T_bat_0 
        self.omega_last = omega_value
        self.Q_heat_last = Q_heat_value
        self.current_last = disturbances[self.current_iterate].item()   

        print(omega_value, Q_heat_value, self.xt_pred[0], T_bat_0, T_bat_target)

        print("total errors", self.total_errors)

        elapsed = 1000*(time.time() - start)
        print(elapsed, 'ms')
        #print("--------------------------------")
        self.current_iterate += 1

        return omega_value, Q_heat_value, T_bat_pred_nn, T_bat_pred, pred_error_nn, pred_error, 0, 0, 0, elapsed, T_bat_dot_model, T_bat_dot_nn, dT_bat_euler, self.nn_on, slack_x
