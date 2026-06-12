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

### Model of battery dynamics and optimization to find steady state
class BatteryDynamics:
    ## Battery dynamics
    def __init__(self): 
        pass
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

        P = cs.vertcat(current_normalized)
        
        
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

        # Heating/cooling power transferred into battery from coolant
        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)
        # Dynamics
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))/model.T_bat_scale
        X_dot_nominal = cs.vertcat(T_bat_dot_model)
        T_bat_dot_function = cs.Function("f_model", [T_bat_normalized, current_normalized, omega_normalized, Q_heat_normalized], 
        [T_bat_dot_model], ["T_bat_normalized", "current_normalized", "omega_normalized", "Q_heat_normalized"], ["ode"])
        f_expl = X_dot_nominal 
        
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
        model.name = "battery_nominal"
        model.T_bat_dot_function = T_bat_dot_function

        return model, constraint 

    def optimization_problem_steady_state(self, dt, T_env):
        ## Steady state selection optimization problem
        T_bat_scale = 100
        omega_scale = 100
        current_scale = 25
        Q_heat_scale = 1000

        # Add scaling to cost so cost is on unscaled T_bat
        Q = np.diag([10*T_bat_scale**2])
        R = np.diag([1,10])   
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

        # Model parameters (same as in BatteryDynamics)
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

        # Fitting parameters (same as in BatteryDynamics)
        alpha_0 = 0.635039 
        alpha_1 = 0.915692  
        alpha_2 = 0.919681   
        alpha_3 = 1.47275 
        gamma = 7.38325 

        # column vector for storing disturbances 
        P = cs.SX.sym('P', n_disturbances)

        # Dynamics (same as in BatteryDynamics)

        NTU_bat  = (alpha_3 * hA_bat) / (mdot_c*c_coolant + 1e-3)
        T_clin= (T_bat + alpha_1*(1/(1-cs.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
        T_clout = ((T_clin - T_bat) * alpha_2*cs.exp(-NTU_bat) + T_bat)
        T_clin_min = -20 + CELSIUS_TO_KELVIN
        T_clin_max = 100 + CELSIUS_TO_KELVIN # Used to be 50
        T_clout_min = -20 + CELSIUS_TO_KELVIN 
        T_clout_max = 100 + CELSIUS_TO_KELVIN
        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma * (T_env - T_bat))/T_bat_scale
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
### Model predictive controller definition
class MPC:
    def __init__(self, model, constraint, N, t_horizon):
        # Model and constraints from BatteryDynamics
        self.model = model
        self.constraint = constraint
        self.N = N
        self.t_horizon = t_horizon

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
        ny = nx + nu # Stage cost variable dimension
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
        ocp.cost.Vz = np.array([[]]) # No algebraic states here
        ocp.cost.Vx_e = np.eye(nx)
      

        # Define weight parameters
        # T_bat is in scaled formulation, so add scaling to cost so it operates on the unscaled T_bat
        Q = np.diag([10*model.T_bat_scale**2])
        R = np.diag([1,10])
        
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
        ocp.solver_options.sim_method_num_stages = 1 # Use explicit euler integration
        
        # Will be overwritten
        ocp.parameter_values = 0

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
        self.N =200 # Prediction horizon
        self.t_horizon = self.N * 5 # In time, remember we sample every 5 s so with N = 200 --> t_horizon = 1000 s
        self.dt = self.t_horizon/self.N # 5 s

        model = BatteryDynamics()
        casadi_model, constraint = model.model(T_env=T_env)
        self.T_bat_dot_function = casadi_model.T_bat_dot_function
        self.solver = MPC(model=casadi_model, constraint=constraint, N=self.N, t_horizon = self.t_horizon).solver # Returns the solver object from MPC
        
        # Time for which simulation is warm started. I.e. for the first 30 s a pre-determined input is used
        self.T_warm_start = 30 # Helps with initializing the Simulink so that the first iterations are not super slow


        self.current_iterate = 0
        self.xt_pred = np.array([CELSIUS_TO_KELVIN])
        self.x_last = np.array([0])
        self.omega_last = 0
        self.Q_heat_last = 0
        self.current_last = 0
        self.data = []
        self.total_errors = 0
        self.total_cost = 0
        self.omega_scale = casadi_model.omega_scale
        self.Q_heat_scale = casadi_model.Q_heat_scale
        self.T_bat_scale = casadi_model.T_bat_scale
        self.current_scale = casadi_model.current_scale
        self.T_env = T_env

        self.residual_dictionary = {'run': [], 'T_bat':[], 'current': [], 'omega_scaled': [], 'Q_heat_scaled': [], 'Q_cool':[], 'residual': [], 'T_bat_dot_model': [], 'T_bat_dot_euler': [] }
        
        # Reading disturbance info
        df = pd.read_csv("current_rms_5s.csv", names=["current"])

        arr = df.to_numpy(dtype=np.float32)
        self.disturbance_values = arr

        self.steady_state_solver, self.steady_state_args = model.optimization_problem_steady_state(dt=self.dt, T_env=T_env)
        
    def get_steady_state(self, T_bat_target):
        # Gets steady state for tracking (so reference on input is not 0, but u_ss)
        args = self.steady_state_args
        
        disturbances = self.disturbance_values
        current_N = disturbances[self.current_iterate + self.N].item()/self.current_scale

        args['p'] = cs.vertcat(
            current_N,
        )
        # Call optimization problem for steady state (defined in BatteryDynamics)
        sol = self.steady_state_solver(
            x0 = args['x0'],
            lbx = args['lbx'],
            ubx = args['ubx'],
            lbg = args['lbg'],
            ubg = args['ubg'],
            p = args['p']
        )
        print(sol['x'])
        omega_norm_ss = sol['x'][2]
        Q_heat_norm_ss = sol['x'][3]

        return omega_norm_ss.full().item(), Q_heat_norm_ss.full().item()

    def get_cost_to_go(self, T_bat_target, T_env, omega_norm_ss, Q_heat_norm_ss):
        # Calculates cost-to-go for recursive feasibility (linearized)
        T_bat_target = T_bat_target/self.T_bat_scale
        T_bat_normalized = cs.MX.sym('T_bat_normalized')
        T_bat = self.T_bat_scale * T_bat_normalized
        omega_norm = cs.MX.sym('omega_norm')
        omega = self.omega_scale * omega_norm
        Q_heat_norm = cs.MX.sym('Q_heat_norm')
        Q_heat = self.Q_heat_scale * Q_heat_norm 

        U = cs.vertcat(omega_norm, Q_heat_norm)
                
        # Parameters (same as in BatteryDynamics)
        m_battery = 20*2.5*4
        c_battery = 795
        c_coolant = 3500
        density_coolant = 1050
                
        pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
        R_battery = 4*20*0.0128 # Battery resistance
        C_battery = 28*3600 # in coloumb
        hA_bat = 2500        

        # Fitting parameters (same as in BatteryDynamics)
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

        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (- Q_cool + gamma*(T_env - T_bat))
        
        # Linearization by calculating Jacobians
        f_T_bat_dot = cs.Function("f_model", [T_bat_normalized,U], [T_bat_dot_model], ["x", "u"], ["ode"])

        jac_T_bat = cs.jacobian(T_bat_dot_model, T_bat_normalized)
        jac_T_bat_fun = cs.Function("f_dT_bat", [T_bat_normalized,U], [jac_T_bat], ["x", "u"], ["f_dT_bat"])

        jac_U = cs.jacobian(T_bat_dot_model, U)
        jac_U_fun = cs.Function("f_dU", [T_bat_normalized,U], [jac_U], ["x", "u"], ["f_dU"])

        A = jac_T_bat_fun(T_bat_target, [omega_norm_ss, Q_heat_norm_ss]).full()
        B = jac_U_fun(T_bat_target, [omega_norm_ss, Q_heat_norm_ss]).full()
        B = B.reshape((1,2))

        a = A
        b = B
        q = 10*self.T_bat_scale**2
        r = np.diag([1,10])
        
        cost_to_go = scipy.linalg.solve_continuous_are(a = a, b = b, q = q, r = r).item()
        Q_e = np.diag([cost_to_go])
        return Q_e, cost_to_go
    def collect_data(self, T_bat_0, dT_bat):
        # Save data to be used in pretraining 

        omega = self.omega_last
        Q_heat = self.Q_heat_last 
        current = self.current_last
        T_bat_k_minus_1 = self.T_bat_last
        T_bat_k = T_bat_0

        # Approximate derivative
        dT_bat_euler = (T_bat_k - T_bat_k_minus_1)/self.dt
        

        
        # Parameters (same as in BatteryDynamics)
        m_battery = 20*2.5*4
        c_battery = 795
        c_coolant = 3500
        density_coolant = 1050
            
        pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
        R_battery = 4*20*0.0128 # Battery resistance
        C_battery = 28*3600 # in coloumb
        hA_bat = 2500

        # Fitting parameters (same as in BatteryDynamics)
        alpha_0 = 0.635039 
        alpha_1 = 0.915692 
        alpha_2 = 0.919681  
        alpha_3 = 1.47275
        gamma = 7.38325

        # Dynamics          
        mdot_c = density_coolant*pump_displacement*omega
  
        NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
        T_clin= (self.T_bat_last + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
        T_clout = ((T_clin - self.T_bat_last) * alpha_2*np.exp(-NTU_bat) + self.T_bat_last)
          
        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(self.T_env - self.T_bat_last))

        # Dynamics residual
        residual = dT_bat_euler - T_bat_dot_model

        self.data.append((self.T_bat_last, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale))
        self.residual_dictionary['run'].append(0)
        self.residual_dictionary['T_bat'].append(self.T_bat_last/self.T_bat_scale)
        self.residual_dictionary['current'].append(current/self.current_scale)
        self.residual_dictionary['Q_cool'].append(Q_cool)
        self.residual_dictionary['omega_scaled'].append(omega/self.omega_scale)
        self.residual_dictionary['Q_heat_scaled'].append(Q_heat/self.Q_heat_scale)
        self.residual_dictionary['residual'].append(residual)
        self.residual_dictionary['T_bat_dot_model'].append(T_bat_dot_model)
        self.residual_dictionary['T_bat_dot_euler'].append(dT_bat_euler)

    def get_input(self, T_bat_target, T_bat_0, SOC_0, current_0, dT_bat, dSOC):
        # MPC step that returns input (and other values for plotting)
        print("--------------------------------")
        print("NOMINAL MPC BATTERY MODEL WITH TARGET TRACKING AND COST-TO-GO")
        T_bat_target = T_bat_target/self.T_bat_scale
        
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
        
        # Get steady state reference for inputs
        omega_norm_ss, Q_heat_norm_ss = self.get_steady_state(T_bat_target=T_bat_target)
        # Get cost-to-go weighting matrix
        Q_e, cost_to_go = self.get_cost_to_go(T_bat_target, T_env, omega_norm_ss, Q_heat_norm_ss)
   
        # Set reference and current (disturbance) for each step in MPC horizon
        for k in range(self.N):
            y_ref_k = np.array([T_bat_target, omega_norm_ss, Q_heat_norm_ss])
            self.solver.set(k, "yref", y_ref_k)
            param_values = disturbances[self.current_iterate + k].item()/self.current_scale
            
            self.solver.set(k, "p", param_values)

        # Set terminal reference
        y_ref_terminal = np.array([T_bat_target]) # Terminal cost only on states 
        self.solver.set(self.N, "yref", y_ref_terminal)
        # Set current on last time step
        param_values = disturbances[self.current_iterate + self.N].item()/self.current_scale

        self.solver.set(self.N, "p", param_values)
        # Set cost-to-go on terminal node
        self.solver.cost_set(self.N, 'W', Q_e)

        start = time.time()
        xt = np.array([T_bat_0/self.T_bat_scale])    

        # Apply current state as constraint
        self.solver.set(0, "lbx", xt)
        self.solver.set(0, "ubx", xt)

        omega_min, omega_max = (150*2*np.pi/60)/self.omega_scale, (4000*2*np.pi/60)/self.omega_scale
        Q_heat_min, Q_heat_max = -4000/self.Q_heat_scale, 4000/self.Q_heat_scale       

        # Solve mpc and apply control
        self.solver.solve()
        
        self.total_cost += self.solver.get_cost()
        # Input
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
        
        print("inputs normalized", ut)
        
        status = self.solver.get_status()
        if status != 0:
            print("ERROR", status, "at iterate", self.current_iterate)
            self.total_errors += 1
        print("total errors", self.total_errors)

        pred_error = 0
        pred_error_nn = 0
        T_bat_pred = 0
        T_bat_pred_nn = 0 # No NN so set to 0 always
        T_bat_dot_model = dT_bat
        T_bat_dot_nn = dT_bat
        dT_bat_euler = dT_bat

        if self.current_iterate  > 0:
            # Calculate prediction error
            omega = self.omega_last
            Q_heat = self.Q_heat_last 
            current = self.current_last
            T_bat_k_minus_1 = self.T_bat_last
            T_bat_k = T_bat_0

            # Approximate derivative
            dT_bat_euler = (T_bat_k - T_bat_k_minus_1)/self.dt
            print("derivative read from model", dT_bat, "derivative euler step", dT_bat_euler)


            # Calculate derivative with model
            # Parameters (same as in BatteryDynamics)
            m_battery = 20*2.5*4
            c_battery = 795
            c_coolant = 3500
            density_coolant = 1050
            
            pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
            R_battery = 4*20*0.0128 # Battery resistance
            C_battery = 28*3600 # in coloumb
            hA_bat = 2500

            # Fitting parameters (same as in BatteryDynamics)
            alpha_0 = 0.635039 
            alpha_1 = 0.915692 
            alpha_2 = 0.919681 
            alpha_3 = 1.47275
            gamma = 7.38325

            # Dynamics
            mdot_c = density_coolant*pump_displacement*omega
            NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
            T_clin= (self.T_bat_last + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
            T_clout = ((T_clin - self.T_bat_last) * alpha_2*np.exp(-NTU_bat) + self.T_bat_last)
          
            Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

            T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(self.T_env - self.T_bat_last))
                
            K1 = cs.DM.full(self.T_bat_dot_function(T_bat_k_minus_1, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale)).item()

            T_bat_pred = T_bat_k_minus_1 + (self.dt ) * (K1) 
            pred_error_nn = T_bat_pred - T_bat_k
            T_bat_pred_nn = T_bat_pred
            T_bat_dot_nn = T_bat_dot_model

        
        # Want to compare prediction at time step 0 with value at time step 1
        # So delay update of xt pred
        if self.current_iterate > 0:
            pred_error = self.xt_pred[0] - T_bat_0/self.T_bat_scale
        else:
            pred_error = 0
        T_bat_pred = self.xt_pred[0]
        pred = self.solver.get(1, 'x')
        
              
        self.xt_pred = np.array([pred[0].item()])
        # Collect data on state, inputs and disturbance now, which in next iteration will be the last applied 
        self.x_last = xt
        self.omega_last = omega_value
        self.Q_heat_last = Q_heat_value
        self.current_last = disturbances[self.current_iterate].item()   

        # Collect data
        if self.current_iterate > 0:
            self.collect_data(T_bat_0, dT_bat)
  
        self.T_bat_last = T_bat_0

        if (self.current_iterate * self.dt) < self.T_warm_start:
            # Warm start (helps with initializing Simulink)
            if T_bat_0/self.T_bat_scale > T_bat_target: # Cooling regime
                omega_value, Q_heat_value = self.omega_scale*omega_max, self.Q_heat_scale*Q_heat_min
            else: # Heating regime
                omega_value, Q_heat_value = self.omega_scale*omega_max, self.Q_heat_scale*Q_heat_max
    
        # Save data to CSV
        if self.dt*self.current_iterate >= 3*2470:
            df = pd.DataFrame(data=self.residual_dictionary)
            df.to_csv("Offline_Training/Training_Data/residuals_matched_heating_T_env_5.csv", index=False)

        elapsed = 1000*(time.time() - start)

        print(elapsed, 'ms')
        print("--------------------------------")
        self.current_iterate += 1
        nn_on = 0 # No neural network
        # Only omega_value and Q_heat_value is needed for control, rest for plotting
        return omega_value, Q_heat_value, T_bat_pred_nn, T_bat_pred, pred_error_nn, pred_error, self.omega_scale*omega_norm_ss, self.Q_heat_scale*Q_heat_norm_ss, cost_to_go, elapsed, T_bat_dot_model, T_bat_dot_nn, dT_bat_euler, nn_on, slack_x