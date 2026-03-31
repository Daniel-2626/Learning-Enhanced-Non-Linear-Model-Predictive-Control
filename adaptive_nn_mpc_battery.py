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

np.random.seed(42)
CELSIUS_TO_KELVIN = 273.15
class MLP(nn.Module):
    def __init__(self, input_dim=2, output_dim=1, hidden_dim=128, num_layers=3):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()] # nn.ReLU rectified linear function (max(x,0))
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]) # nn.Linear applies an affine transform. hidden_dim features and hidden_dim out features
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

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
        SOC= cs.MX.sym('SOC')
        X = cs.vertcat(T_bat, SOC)

        omega_normalized = cs.MX.sym('omega_norm')
        omega = model.omega_scale * omega_normalized #cs.MX.sym('omega') # Power into heater/cooler given as efficiency*Pin
        Q_heat_normalized = cs.MX.sym('Q_heat_norm')
        Q_heat = model.Q_heat_scale * Q_heat_normalized #cs.MX.sym('Q_heat') # Pump control (rpm that is converted to kg/s)
        current = cs.MX.sym('current')
        nn_on = cs.MX.sym('nn_on') # Switch for when the NN is on

        P = cs.vertcat(current, nn_on)
                
        U = cs.vertcat(omega_normalized, Q_heat_normalized)
        
        nx = 2
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

        

        #[0.637968, 1, 0.980682, 0.981036]
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
        
        constraint = cs.types.SimpleNamespace()
        constraint.T_clin_min = -20 + CELSIUS_TO_KELVIN 
        constraint.T_clin_max = 100 + CELSIUS_TO_KELVIN
        constraint.T_clout_min = -20 + CELSIUS_TO_KELVIN
        constraint.T_clout_max = 100 + CELSIUS_TO_KELVIN
        constraint.expr = cs.vertcat(T_clin, T_clout)

        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        # Now for the actual calculations 
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))
        SOC_dot_model = -current/C_battery
        X_dot_nominal = cs.vertcat(T_bat_dot_model, SOC_dot_model)

        # MLP network
        mlp_input = cs.vertcat(T_bat, current, U)
        residual = self.residual_model(mlp_input.T).T 
        X_dot_residual = cs.vertcat(residual[0], 0) # x1 dot and x2 dot residual
        
        f_expl = X_dot_nominal + nn_on*X_dot_residual
        x_start = np.array([T_env,1]) # initial constraint (gets overwritten)

        # store to struct
        
        model.x = X 
        model.xdot = cs.MX.sym('xdot', 2)
        model.u = U
        model.z = cs.vertcat([])
        model.p = P
        model.f_expl = f_expl
        model.f_nominal = X_dot_nominal
        model.x_start = x_start
        model.constraints = cs.vertcat([]) # add constraints here or in mpc?
        model.name = "battery"

        
        return model, constraint 
    def optimization_problem_steady_state(self, t_horizon, T_env):
        # Works in normalized 
        Q = np.diag([10])
        R = np.diag([1, 1])   
        T = np.diag([1000000,1000000])


        N = 1 # number of look ahead steps

        t_horizon = t_horizon 
        t_env = T_env
        step_horizon = t_horizon/N # time between steps in seconds

        # Constraints
        T_bat_max = 50 + CELSIUS_TO_KELVIN
        T_bat_min = -20 + CELSIUS_TO_KELVIN

        # State symbolic variables
        T_bat = cs.SX.sym('T_bat')
        slack_RK4 = cs.SX.sym("Slack RK4")
        slack_ss = cs.SX.sym("Slack ss")
        S = cs.vertcat(slack_ss, slack_RK4)
        n_slack = S.numel()
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

        disturbances = cs.vertcat(current)
        n_disturbances = disturbances.numel()

        # matrix containing all states over all time steps
        X = cs.SX.sym('X', n_states, N+1)

        # matrix containing all control actions over all time steps
        U = cs.SX.sym('U', n_controls, N)

        T_env = T_env
        T_bat_target = 20.5 + CELSIUS_TO_KELVIN

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
        alpha_0 = 0.635039 # 0.637968 
        alpha_1 = 0.915692 # 1 
        alpha_2 = 0.919681 # 0.980682  
        alpha_3 = 1.47275 # 0.981036
        gamma = 7.38325 # 7.59853

        # column vector for storing disturbances 
        P = cs.SX.sym('P', n_disturbances)

        # Get the cooler in and out temps
        NTU_bat  = (alpha_3 * hA_bat) / (mdot_c*c_coolant + 1e-3)
        T_clin= (T_bat + alpha_1*(1/(1-cs.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
        T_clout = ((T_clin - T_bat) * alpha_2*cs.exp(-NTU_bat) + T_bat)
        T_clin_min = -20 + CELSIUS_TO_KELVIN
        T_clin_max = 50 + CELSIUS_TO_KELVIN
        T_clout_min = -20 + CELSIUS_TO_KELVIN 
        T_clout_max = 50 + CELSIUS_TO_KELVIN
        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        # Now for the actual calculations 
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma * (T_env - T_bat))
        ode_model = T_bat_dot_model

        f_model = cs.Function("f_model", [states,controls,disturbances], [ode_model], ["x", "u", "d"], ["ode"])
        T_clin_fun = cs.Function("T_clin_fun", [states,controls], [T_clin], ["x", "u"], ["T_clin_fun"])
        T_clout_fun = cs.Function("T_clout_fun", [states,controls], [T_clout], ["x", "u"], ["T_clout_fun"])

        cost_fn = 0 # Cost function
        g = X[:, 0] - T_bat_target # constraints in the equation, that x0 = p0 

        st = X[:, 0]
        disturbances = P[-n_disturbances]

        cost_fn = (U).T @ R @ (U) + S.T @ T @ S + (X[:, 1]-X[:,0]).T @ Q @ (X[:, 1] -X[:,0])
        T_clin_eval = T_clin_fun(st, U)
        T_clout_eval = T_clout_fun(st, U)

        st_next = X[:, 1]

        K1 = f_model(st, U, disturbances)
        K2 = f_model(st + step_horizon/2 * K1, U, disturbances)
        K3 = f_model(st + step_horizon/2 * K2, U, disturbances)
        K4 = f_model(st + step_horizon * K3, U, disturbances)

        st_next_RK4 = st + (step_horizon) * (K1+ 2*K2 + 2*K3 + K4)

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
            'acceptable_obj_change_tol': 1e-6
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

        lbx[0: n_states*(N+1)] = T_bat_min # Different notation to matlab. Here have [start:stop.step]. We fill out the vector related to states [start:n_states*(N+1)] with a lower bound every n_states, meaning we set lower bound for first state at each time step
        lbx[-4] = omega_min
        lbx[-3] = Q_heat_min
        lbx[-2:] = -100
        ubx[0: n_states*(N+1)] = T_bat_max 
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
            'lbg': lg, # constrainst lower bound (basically giving == constraint)
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
        nx = 2
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
        ocp.cost.Vx = np.zeros((ny, nx)) # i feel like this should be nx,nx ||Vx x||^2 
        for i in range(nx):
            ocp.cost.Vx[i,i] = 1 # so set the matrix coeff corr to u to 0. maybe the dim of Vx is set to be the same as Vu
        ocp.cost.Vu = np.zeros((ny, nu))
        for i in range(nu):
            ocp.cost.Vu[i + nx, i] = 1
        ocp.cost.Vz = np.array([[]]) # don't know what the V_z z, what the variable z should be
        ocp.cost.Vx_e = np.eye(nx)
        l4c_y_expr = None

        # Define weight parameters
        Q = np.diag([10, 0.1])
        R = np.diag([1, 1])
        ocp.cost.W = scipy.linalg.block_diag(Q,R)
        ocp.cost.W_e = Q 
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # Initial state
        ocp.constraints.x0 = model.x_start

        # Set constraints
        omega_max = (4000*2*np.pi/60)/model.omega_scale
        
        omega_min = (150*2*np.pi/60)/model.omega_scale
        print("omega max", omega_max)
    
        Q_heat_max = 4000/model.Q_heat_scale
        print("Q_heat max", Q_heat_max)
        Q_heat_min = -4000/model.Q_heat_scale
        Tb_max = 50 + CELSIUS_TO_KELVIN
        Tb_min = -20 + CELSIUS_TO_KELVIN
        SOC_max = 1
        SOC_min = 0
        ocp.constraints.lbu = np.array([omega_min, Q_heat_min])
        ocp.constraints.ubu = np.array([omega_max, Q_heat_max])

        ocp.constraints.idxbu = np.array([0,1])
        ocp.constraints.idxbx = np.array([0]) # at what indices to have constraints
        ocp.constraints.ubx = np.array([Tb_max])
        ocp.constraints.lbx = np.array([Tb_min])
        
        # Maybe will complain that constraint is a parameter
        ocp.constraints.lh = np.array(
        [
            model.T_env,
            model.T_env
        ]
        )
        ocp.constraints.uh = np.array(
            [
                constraint.T_clin_max,
                constraint.T_clout_max
            ]
        )
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
        ocp.solver_options.model_external_shared_lib_dir = self.external_shared_lib_dir
        ocp.solver_options.model_external_shared_lib_name = self.external_shared_lib_name 
        
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



class Controller:
    def setup(self, T_bat_target, T_env):
        
        # Residual MLP: Lightweight
        residual_mlp = MLP(input_dim = 2 + 2, output_dim=1, hidden_dim=64, num_layers=3) # the network
        for param in residual_mlp.parameters():
            param.requires_grad = False
        self.residual_mlp = residual_mlp
        self.residual_optimizer = torch.optim.Adam(residual_mlp.parameters(), lr=1e-4) # lr = learning rate, the optimizer
        self.residual_criterion = nn.MSELoss()

        l4c_residual = l4c.L4CasADi(self.residual_mlp, name="battery", mutable=True)
        self.l4c_residual = l4c_residual

        # MPC Setup 
        self.N = 80
        self.t_horizon = self.N*5
        learned_model = BatteryLearnedDynamics(l4c_residual)
      
        casadi_model, constraint = learned_model.model(T_env=T_env)
        
        # MIGHT CAUSE ISSUES SINCE DIFFERENT THAN YU MEI, here model=casadi_model
        self.solver = MPC(model=casadi_model, constraint=constraint, N=self.N, t_horizon = self.t_horizon,
                    external_shared_lib_dir=l4c_residual.shared_lib_dir,
                    external_shared_lib_name=l4c_residual.name).solver # Returns the solver object from MPC

        # SOC_ref = None # This is not actually tracked
        self.dt = self.t_horizon/self.N
        self.obs_buffer = []
        self.batch_size = 30
        self.T_update = 60 
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

        self.steady_state_solver, self.steady_state_args = learned_model.optimization_problem_steady_state(t_horizon=self.t_horizon, T_env=T_env)
        
    def get_steady_state(self, T_bat_target):
        
        args = self.steady_state_args
        
        disturbances = self.disturbance_values
        current_0 = disturbances[int(self.dt*(self.current_iterate ))].item()
        args['p'] = cs.vertcat(
            current_0,
        )
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

        return omega_norm_ss.full().item(), Q_heat_norm_ss.full().item()
    # Calculates cost-to-go for recursive feasibility (linearized)
    def get_cost_to_go(self, T_bat_target, T_env, omega_norm_ss, Q_heat_norm_ss):
        T_bat = cs.MX.sym('T_bat')
        omega_norm = cs.MX.sym('omega_norm')
        omega = self.omega_scale * omega_norm
        Q_heat_norm = cs.MX.sym('Q_heat_norm')
        Q_heat = self.Q_heat_scale * Q_heat_norm #cs.MX.sym('Q_heat')

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

        # Now for the actual calculations 
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (- Q_cool + gamma*(T_env - T_bat))
        f_T_bat_dot = cs.Function("f_model", [T_bat,U], [T_bat_dot_model], ["x", "u"], ["ode"])

        jac_T_bat = cs.jacobian(T_bat_dot_model, T_bat)
        jac_T_bat_fun = cs.Function("f_dT_bat", [T_bat,U], [jac_T_bat], ["x", "u"], ["f_dT_bat"])

        jac_U = cs.jacobian(T_bat_dot_model, U)
        jac_U_fun = cs.Function("f_dU", [T_bat,U], [jac_U], ["x", "u"], ["f_dU"])

        A = jac_T_bat_fun(T_bat_target+273.15, [omega_norm_ss, Q_heat_norm_ss]).full()
        B = jac_U_fun(T_bat_target+273.15, [omega_norm_ss, Q_heat_norm_ss]).full()
        B = B.reshape((1,2))

        a = A
        b = B
        q = 10
        r = np.diag([1,1])
        cost_to_go = scipy.linalg.solve_continuous_are(a = a, b = b, q = q, r = r).item()
        Q_e = np.diag([cost_to_go, 1])
        return Q_e
    def get_input(self, T_bat_target, T_bat_0, SOC_0, current_0, dT_bat, dSOC):
       
        print("--------------------------------")
        print("ADAPTIVE ACADOS MPC BATTERY MODEL WITH TARGET TRACKING AND COST-TO-GO")
        start = time.time()

        if self.current_iterate == 0:
            self.xt_pred = ([T_bat_0, SOC_0])
        disturbances = self.disturbance_values
        T_env = self.T_env
        # Set reference for each step in MPC horizon
        #print(disturbances[0], current_0)
        omega_norm_ss, Q_heat_norm_ss = self.get_steady_state(T_bat_target=T_bat_target)
        Q_e = self.get_cost_to_go(T_bat_target, T_env, omega_norm_ss, Q_heat_norm_ss)
        print("omega_ss", omega_norm_ss, "Q_heat_ss", Q_heat_norm_ss)
        print("Q_e", Q_e)
        if self.nn_on:
            print("NN on from hereafter")
        for k in range(self.N):
            #if k == 0: # only store first reference
                #Tb_ref_history.append(Tb_ref)
                # SOC_ref_history.append(SOC_ref)
            # Set terminal reference
            # y_ref_k = np.array([Tb_ref, SOC_ref, 0])
            y_ref_k = np.array([T_bat_target, 0.0, omega_norm_ss, Q_heat_norm_ss])
         
            self.solver.set(k, "yref", y_ref_k)
            # Potential for error
            # T_env must be in Kelvin
            param_values = np.array([disturbances[int(self.dt*(self.current_iterate + k))].item(), self.nn_on])
            self.solver.set(k, "p", param_values)

        # Set terminal reference
        # y_ref_terminal = np.array([Tb_ref, SOC_ref])
        y_ref_terminal = np.array([T_bat_target, 0.0]) # Terminal cost only on states so 2x1 instead of 4x1 above
        #print(y_ref_terminal)
        self.solver.set(self.N, "yref", y_ref_terminal)
        param_values = np.array([disturbances[int(self.dt*(self.current_iterate + self.N))].item(), self.nn_on])

        self.solver.set(self.N, "p", param_values)

        self.solver.cost_set(self.N, 'W', Q_e)

        xt = np.array([T_bat_0,SOC_0])
        #self.solver.set(0, "x", xt)
        

        # Apply current state as constraint
        self.solver.set(0, "lbx", xt)
        self.solver.set(0, "ubx", xt)

        # Solve mpc and apply control
        self.solver.solve()
        self.total_cost += self.solver.get_cost()
        # ut = solver.get(0, "u").item()
        ut = self.solver.get(0, "u")
        slack_lower = self.solver.get(0, "sl")
        slack_upper = self.solver.get(0, "su")
        print("slack lower", slack_lower, "slack upper", slack_upper)
        
        # Saturation
        omega_max = (4000*2*np.pi/60)
        omega_min = (150*2*np.pi/60)
        Q_heat_max = 4000
        Q_heat_min = -4000
        omega_value = min(self.omega_scale * ut[0].item(), omega_max)
        Q_heat_value = min(self.Q_heat_scale * ut[1].item(), Q_heat_max)
        omega_value = max(omega_value, omega_min)
        Q_heat_value = max(Q_heat_value, Q_heat_min)
        
        print("inputs normalized", ut)
        status = self.solver.get_status()
        if status != 0:
            print("ERROR", status, "at iterate", self.current_iterate)
            self.total_errors += 1
    
        # Want to compare prediction at time step 0 with value at time step 1
        # So delay update of xt pred
        if self.current_iterate > 0:
            pred_error = self.xt_pred[0] - T_bat_0
        else:
            pred_error = 0
        pred = self.solver.get(1, 'x')

        # Collect training data for NN

        
        if self.current_iterate  > 0:

            # input and disturbance values (not symbolics because know what happened)
            # Not sure if should take current values or last, but figure that at this moment the change is happening because of the last values
            omega = self.omega_last
            Q_heat = self.Q_heat_last 
            current = self.current_last
            T_bat_k_minus_1 = self.T_bat_last
            T_bat_k = T_bat_0

            # Approximate derivative
            dT_bat_euler = (T_bat_k - T_bat_k_minus_1)/self.dt
            print("derivative read from model", dT_bat, "derivative euler step", dT_bat_euler)


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
            T_clin= (self.T_bat_last + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
            T_clout = ((T_clin - self.T_bat_last) * alpha_2*np.exp(-NTU_bat) + self.T_bat_last)
          
            Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

            T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(self.T_env - self.T_bat_last))

            self.obs_buffer.append((dT_bat_euler, T_bat_dot_model))
            self.data.append((self.T_bat_last, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale))

        else:
            self.obs_buffer.append((0, 0))
            self.data = [(0,0,0,0)]

        # update every T_update = 60 
        # T_warm_start = 30
        if (self.dt*self.current_iterate) > self.T_warm_start and ((self.dt*self.current_iterate) % self.T_update) == 0 and len(self.obs_buffer) >= self.batch_size:
            self.nn_on = 1
            data = np.array(self.data[-self.batch_size:])
            obs = np.array(self.obs_buffer)
            # data[:, :3] h1, h2 and u
            X_batch = torch.tensor(data[:, :], dtype=torch.float32)
            # h1_dot, h2_dot
            y_true = obs[-self.batch_size:, 0]
            y_true = y_true.reshape((-1,1))
        
            

            y_nominal = obs[-self.batch_size:, 1]
            y_nominal = y_nominal.reshape((-1,1))

    
            y_target = torch.tensor(y_true - y_nominal, dtype=torch.float32)
            for p in self.residual_mlp.parameters(): p.requires_grad = True
            for _ in range(50):
                self.residual_optimizer.zero_grad() # optimizer object
                prediction = self.residual_mlp(X_batch) # gives data to network to make a prediction
                loss = self.residual_criterion(prediction, y_target)
                loss.backward() # calculates gradient
                self.residual_optimizer.step() # one optimization step to update parameters
            for p in self.residual_mlp.parameters(): p.requires_grad = False
            print("UPDATE MODEL AT", self.dt*self.current_iterate)
            
            self.l4c_residual.update(self.residual_mlp)



        self.xt_pred = np.array([pred[0].item(), pred[1].item()])
        self.T_bat_last = xt[0].item()
        self.omega_last = omega_value
        self.Q_heat_last = Q_heat_value
        self.current_last = current_0

 
        elapsed = 1000*(time.time() - start)
        print("time", self.dt*self.current_iterate)
        print("total errors", self.total_errors)
    
        self.current_iterate += 1
        if (self.current_iterate * self.dt) < self.T_warm_start:
            omega_value, Q_heat_value = 4000*2*np.pi/60, 4000
        
        print(omega_value, Q_heat_value, self.xt_pred[0], T_bat_0, T_bat_target)
        print("cumulative cost", self.total_cost)
        print(elapsed, 'ms')
        print("--------------------------------")
        return omega_value, Q_heat_value, self.xt_pred[0], pred_error