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

class BatteryDynamics:
    def __init__(self): # remove gym_env for cascaded tank
        pass
    def model(self, T_env): # remove gym_env for cascaded tank
        model = cs.types.SimpleNamespace()
        model.omega_scale = 100
        model.Q_heat_scale = 1000
        model.T_env = T_env
        T_bat = cs.MX.sym('T_bat')
        #SOC= cs.MX.sym('SOC')
        #X = cs.vertcat(T_bat, SOC)

        X = cs.vertcat(T_bat)

        omega_normalized = cs.MX.sym('omega_norm')
        omega = model.omega_scale * omega_normalized #cs.MX.sym('omega') # Power into heater/cooler given as efficiency*Pin
        Q_heat_normalized = cs.MX.sym('Q_heat_norm')
        Q_heat = model.Q_heat_scale * Q_heat_normalized #cs.MX.sym('Q_heat') # Pump control (rpm that is converted to kg/s)
        current = cs.MX.sym('current')

        P = cs.vertcat(current)
        
        #nn_on = cs.MX.sym('nn_on') # Flip switch for whether on not to have the NN in the model (helps when NN not trained yet)
        
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
        hA_bat = 2500 # 2500

        

        # Identified
        alpha_0 = 0.635039 # 0.637968 
        alpha_1 = 0.915692 # 1 
        alpha_2 = 0.919681 # 0.980682  
        alpha_3 = 1.47275 # 0.981036
        gamma = 7.38325 # 7.
        # Arbitrary
        #alpha_0 = 0.3 # 1.4 # arbitrary
        #alpha_1 = 0.1 # 1 
        #alpha_2 = 2 # 0.980682   
        #alpha_3 = 2 # 0.981036
        #gamma = 5 # 7.
        
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
        #SOC_dot_model = -current/C_battery
        X_dot_nominal = cs.vertcat(T_bat_dot_model)
        T_bat_dot_function = cs.Function("f_model", [T_bat, current, omega_normalized, Q_heat_normalized], 
        [T_bat_dot_model], ["T_bat", "current", "omega_normalized", "Q_heat_normalized"], ["ode"])

        f_expl = X_dot_nominal 
        #x_start = np.array([T_env,1]) # initial constraint (gets overwritten)
        x_start = np.array([T_env]) # initial constraint (gets overwritten)

        # Power function (for cost)
        Power =  (295.1748  * mdot_c**2 - 187.6638 * mdot_c + 18.1336)/(model.omega_scale**2) + (Q_heat**2)/(model.Q_heat_scale**2) 
        model.cost_y_expr_0 = cs.vertcat(Power)
        model.cost_y_expr = cs.vertcat(Power)
        model.cost_y_expr_e = cs.vertcat(0.0)
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
        model.name = "battery_nominal"
        model.T_bat_dot_function = T_bat_dot_function

        
        return model, constraint 
    """
    def optimization_problem_steady_state(self, dt, T_env):
        # Works in normalized 
        Q = np.diag([10])
        R = np.diag([1,1])   
        
        #R = np.diag([1, 155000])   
        T = np.diag([1000000,1000000])


        N = 1 # number of look ahead steps

        t_env = T_env
        step_horizon = dt # time between steps in seconds

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
        # CHANGE HERE
        Q_heat = Q_heat_scale * Q_heat_norm
        pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
        mdot_c = density_coolant*pump_displacement*omega
        R_battery = 4*20*0.0128 # Battery resistance
        C_battery = 28*3600 # in coloumb
        hA_bat = 2500
        # Identified
        alpha_0 = 0.635039 # 0.637968 
        alpha_1 = 0.915692 # 1 
        alpha_2 = 0.919681 # 0.980682  
        alpha_3 = 1.47275 # 0.981036
        gamma = 7.38325 # 7.
        # Arbitrary
        #alpha_0 = 0.3 # 1.4 # arbitrary
        #alpha_1 = 0.1 #0.8 # 1 
        #alpha_2 = 2 # 0.980682  
        #alpha_3 = 2 # 0.981036
        #gamma = 5 # 7.
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
    

        # cost_fn = (U).T @ R @ (U) + S.T @ T @ S + (X[:, 1]-X[:,0]).T @ Q @ (X[:, 1] -X[:,0])

        # Calculate power as a function of omega - fit determined polyfit
        density_coolant = 1050
        pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink

        # Loop over the horizon (though N=1, still good practice)
        for k in range(N):
            # Extract the control inputs for time step k from the U matrix
            u_k = U[:, k]  # This gives [omega_norm; Q_heat_norm] for step k
            
            # Now calculate power using the actual decision variable
            omega_norm_k = u_k[0]  # Extract omega_norm for this step
            Q_heat_norm_k = u_k[1]  # Extract Q_heat_norm for this step (if needed)
            
            # Calculate power with the actual decision variable
            mdot_c = density_coolant * pump_displacement * omega_scale * omega_norm_kkil
            Q_heat_k = Q_heat_scale * Q_heat_norm_k
            Power_k = (295.1748 * mdot_c**2 - 187.6638 * mdot_c + 18.1336)/(omega_scale**2) + (Q_heat_k**2)/(Q_heat_scale**2)
            
            # Add to total cost (if you have multiple steps, you'd sum them)
            cost_fn = cost_fn + Power_k

        T_clin_eval = T_clin_fun(st, U)
        T_clout_eval = T_clout_fun(st, U)

        st_next = X[:, 1]

        K1 = f_model(st, U, disturbances)
        K2 = f_model(st + step_horizon/2 * K1, U, disturbances)
        K3 = f_model(st + step_horizon/2 * K2, U, disturbances)
        K4 = f_model(st + step_horizon * K3, U, disturbances)

        st_next_RK4 = st + (step_horizon) * (K1 + 2*K2 + 2*K3 + K4)

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
        print("omega max", omega_max)
            
        Q_heat_max = 4000/Q_heat_scale
        print("Q_heat max", Q_heat_max)
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
        print(ubx)
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
    """
class MPC:
    def __init__(self, model, constraint, N, t_horizon):
        self.model = model
        self.constraint = constraint
        self.N = N
        self.t_horizon = t_horizon

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
        ny = 1
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
        Q = np.array([[1.0]])
        
        # ocp.cost.W = scipy.linalg.block_diag(Q,R)
        # ocp.cost.W_e = Q 
        ocp.cost.W_0 = Q
        ocp.cost.W = Q
        ocp.cost.W_e = Q

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
        
        # Nonlinear constraints (for coolan)
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
        

        # Define slack for terminal states
        ocp.cost.zl_e = 100000 * np.zeros((ns_e,))
        ocp.cost.zu_e = 100000 * np.zeros((ns_e,))
        ocp.cost.Zl_e = 10000000 * np.ones((ns_e,))
        ocp.cost.Zu_e = 10000000 * np.ones((ns_e,))

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
        ocp.cost.zl = 100000 * np.zeros((ns,))
        ocp.cost.zu = 100000 * np.zeros((ns,))
        ocp.cost.Zl = 10000000 * np.ones((ns,))
        ocp.cost.Zu = 10000000 * np.ones((ns,))

        ocp.cost.zl_0 = 100000 * np.zeros((nsh+nsu,))
        ocp.cost.zu_0 = 100000 * np.zeros((nsh+nsu,))
        ocp.cost.Zl_0 = 100000 * np.ones((nsh+nsu,))
        ocp.cost.Zu_0 = 100000 * np.ones((nsh+nsu,))

        # Solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        ocp.solver_options.sim_method_num_stages = 1
        
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
        model_ac.cost_y_expr = model.cost_y_expr
        model_ac.cost_y_expr_0 = model.cost_y_expr_0
        model_ac.cost_y_expr_e = model.cost_y_expr_e

        return model_ac



class Controller:
    def setup(self, T_bat_target, T_env):

        # MPC Setup 
        self.N =200 #200 # 80
        self.t_horizon = self.N * 5
        model = BatteryDynamics()
      
        casadi_model, constraint = model.model(T_env=T_env)
        self.T_bat_dot_function = casadi_model.T_bat_dot_function
        # MIGHT CAUSE ISSUES SINCE DIFFERENT THAN YU MEI, here model=casadi_model
        self.solver = MPC(model=casadi_model, constraint=constraint, N=self.N, t_horizon = self.t_horizon).solver # Returns the solver object from MPC

        # SOC_ref = None # This is not actually tracked
        self.dt = self.t_horizon/self.N
        self.obs_buffer = []
        self.batch_size = self.N # at most self.N - self.T_warm_start at least
        self.T_update = 60 
        self.T_warm_start = 30
        self.current_iterate = 0
        self.xt_pred = np.array([CELSIUS_TO_KELVIN, 1])
        self.x_last = np.array([0,0])
        self.omega_last = 0
        self.Q_heat_last = 0
        self.current_last = 0
        self.data = []
        self.total_errors = 0
        self.total_cost = 0
        self.omega_scale = casadi_model.omega_scale
        self.Q_heat_scale = casadi_model.Q_heat_scale
        self.T_env = T_env

        #self.residual_dictionary = {'run': [], 'T_bat':[], 'current': [], 'omega_scaled':[], 'Q_heat_scaled':[], 'residual': []}
        self.residual_dictionary = {'run': [], 'T_bat':[], 'current': [], 'omega_scaled': [], 'Q_heat_scaled': [], 'Q_cool':[], 'residual': []}
        
        # Reading disturbance info
        df = pd.read_csv("current_intp1.csv", names=["current"])
        arr = df.to_numpy(dtype=np.float32)
        self.disturbance_values = arr

        #self.steady_state_solver, self.steady_state_args = model.optimization_problem_steady_state(dt=self.dt, T_env=T_env)
    """    
    def get_steady_state(self, T_bat_target):
        
        args = self.steady_state_args
        
        disturbances = self.disturbance_values
        current_0 = disturbances[int(self.dt*(self.current_iterate + self.N))].item()
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
        print(sol['x'])
        omega_norm_ss = sol['x'][2]
        Q_heat_norm_ss = sol['x'][3]

        return omega_norm_ss.full().item(), Q_heat_norm_ss.full().item()
    """
    # # Calculates cost-to-go for recursive feasibility (linearized)
    # def get_cost_to_go(self, T_bat_target, T_env, omega_norm_ss, Q_heat_norm_ss):
    #     T_bat = cs.MX.sym('T_bat')
    #     omega_norm = cs.MX.sym('omega_norm')
    #     omega = self.omega_scale * omega_norm
    #     Q_heat_norm = cs.MX.sym('Q_heat_norm')
    #     Q_heat = self.Q_heat_scale * Q_heat_norm #cs.MX.sym('Q_heat')

    #     U = cs.vertcat(omega_norm, Q_heat_norm)
                
    #     # Parameters
    #     m_battery = 20*2.5*4
    #     c_battery = 795
    #     c_coolant = 3500
    #     density_coolant = 1050
                
    #     pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
    #     R_battery = 4*20*0.0128 # Battery resistance
    #     C_battery = 28*3600 # in coloumb
    #     hA_bat = 2500        

    #     # Identified
    #     alpha_0 = 0.635039 # 0.637968 
    #     alpha_1 = 0.915692 
    #     alpha_2 = 0.919681 
    #     alpha_3 = 1.47275 # 0.981036
    #     gamma = 7.38325 # 7.
    #     # Arbitrary
    #     #alpha_0 = 0.3 # 1.4 # arbitrary
    #     #alpha_1 = 0.1 # 1 
    #     #alpha_2 = 2 # 0.980682  
    #     #alpha_3 = 2 # 0.981036
    #     #gamma = 5 # 7.
                
    #     mdot_c = density_coolant*pump_displacement*omega
    #     # Dynamics
                
    #     # Get the cooler in and out temps
    #     NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
    #     T_clin= (T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
    #     T_clout = ((T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat)
    #     Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

    #     # Now for the actual calculations 
    #     T_bat_dot_model = alpha_0/(m_battery*c_battery) * (- Q_cool + gamma*(T_env - T_bat))
    #     f_T_bat_dot = cs.Function("f_model", [T_bat,U], [T_bat_dot_model], ["x", "u"], ["ode"])

    #     jac_T_bat = cs.jacobian(T_bat_dot_model, T_bat)
    #     jac_T_bat_fun = cs.Function("f_dT_bat", [T_bat,U], [jac_T_bat], ["x", "u"], ["f_dT_bat"])

    #     jac_U = cs.jacobian(T_bat_dot_model, U)
    #     jac_U_fun = cs.Function("f_dU", [T_bat,U], [jac_U], ["x", "u"], ["f_dU"])

    #     A = jac_T_bat_fun(T_bat_target+273.15, [omega_norm_ss, Q_heat_norm_ss]).full()
    #     B = jac_U_fun(T_bat_target+273.15, [omega_norm_ss, Q_heat_norm_ss]).full()
    #     B = B.reshape((1,2))

    #     a = A
    #     b = B
    #     q = 10
    #     r = np.diag([1,1])
        
    #     #r = np.diag([1,155000])
    #     cost_to_go = scipy.linalg.solve_continuous_are(a = a, b = b, q = q, r = r).item()
    #     Q_e = np.diag([cost_to_go, 1])
    #     return Q_e, cost_to_go
    

    def collect_data(self, T_bat_0, dT_bat):
        # input and disturbance values (not symbolics because know what happened)
        # Not sure if should take current values or last, but figure that at this moment the change is happening because of the last values
        omega = self.omega_last
        Q_heat = self.Q_heat_last 
        current = self.current_last
        T_bat_k_minus_1 = self.T_bat_last
        T_bat_k = T_bat_0

        # Approximate derivative
        dT_bat_euler = (T_bat_k - T_bat_k_minus_1)/self.dt
        

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
        residual = dT_bat_euler - T_bat_dot_model

        self.obs_buffer.append((dT_bat_euler, T_bat_dot_model))
        self.data.append((self.T_bat_last, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale))
        self.residual_dictionary['run'].append(0)
        self.residual_dictionary['T_bat'].append(self.T_bat_last)
        self.residual_dictionary['current'].append(current)
        self.residual_dictionary['Q_cool'].append(Q_cool)
        self.residual_dictionary['omega_scaled'].append(omega/self.omega_scale)
        self.residual_dictionary['Q_heat_scaled'].append(Q_heat/self.Q_heat_scale)
        self.residual_dictionary['residual'].append(residual)

    def get_input(self, T_bat_target, T_bat_0, SOC_0, current_0, dT_bat, dSOC):
       
        print("--------------------------------")
        print("NOMINAL ACADOS MPC BATTERY MODEL WITH TARGET TRACKING AND COST-TO-GO")
        if self.current_iterate == 0:
            #self.xt_pred = ([T_bat_0, SOC_0])
            self.xt_pred = ([T_bat_0])
            # Warm start
            omega_warm_start = (2000*2*np.pi/60)/self.omega_scale
            if T_bat_0 <= 20.5:
                Q_heat_warm_start = 2000/self.Q_heat_scale
            else:
                Q_heat_warm_start = -2000/self.Q_heat_scale
            input_warm_start = np.array([omega_warm_start, Q_heat_warm_start])
            self.solver.set(0, 'u', input_warm_start) # Weird but this is how acados does warm starts. Does not "force" inputs to be this
            #state_warm_start = np.array([T_bat_0, SOC_0])
            state_warm_start = np.array([T_bat_0])
            self.solver.set(0, 'x', state_warm_start)
        disturbances = self.disturbance_values
        T_env = self.T_env

        # Set reference for each step in MPC horizon
        #print(disturbances[0], current_0)
        #omega_norm_ss, Q_heat_norm_ss = self.get_steady_state(T_bat_target=T_bat_target)

        # Q_e, cost_to_go = self.get_cost_to_go(T_bat_target, T_env, omega_norm_ss, Q_heat_norm_ss)
        # print("omega_ss", omega_norm_ss, "Q_heat_ss", Q_heat_norm_ss)
        # print("Q_e", Q_e)

        # for k in range(self.N):
        #     #if k == 0: # only store first reference
        #         #Tb_ref_history.append(Tb_ref)
        #         # SOC_ref_history.append(SOC_ref)
        #     # Set terminal reference
        #     # y_ref_k = np.array([Tb_ref, SOC_ref, 0])
        #     y_ref_k = np.array([T_bat_target, 0.0, omega_norm_ss, Q_heat_norm_ss])
        #     self.solver.set(k, "yref", y_ref_k)
        #     # Potential for error
        #     # T_env must be in Kelvin
            
        #     param_values = disturbances[int(self.dt*(self.current_iterate + k))].item()
        #     self.solver.set(k, "p", param_values)

        # # Set terminal reference
        # # y_ref_terminal = np.array([Tb_ref, SOC_ref])
        # y_ref_terminal = np.array([T_bat_target, 0.0]) # Terminal cost only on states so 2x1 instead of 4x1 above
        # #print(y_ref_terminal)
        # self.solver.set(self.N, "yref", y_ref_terminal)
        # param_values = disturbances[int(self.dt*(self.current_iterate + self.N))].item()
        # self.solver.set(self.N, "p", param_values)

        # self.solver.cost_set(self.N, 'W', Q_e)

        for k in range(self.N):
            # For stage cost: [T_bat_target, Power_target]
            # Power_target = 0 to minimize power consumption
            y_ref_k = np.array([0])  
            self.solver.set(k, "yref", y_ref_k)
            
            # Set disturbances (current)
            #param_values = disturbances[int(self.dt*(self.current_iterate + k))].item()
            param_values = 0

            self.solver.set(k, "p", param_values)

        y_ref_terminal = np.array([0])  
        self.solver.set(self.N, "yref", y_ref_terminal)
        #param_values = disturbances[int(self.dt*(self.current_iterate + self.N))].item()
        param_values = 0
        self.solver.set(self.N, "p", param_values)

        #Q_e = 10
        #self.solver.cost_set(self.N, 'W', np.array([[Q_e]]))  # Q_e should be scalar

        start = time.time()
        #xt = np.array([T_bat_0,SOC_0])
        xt = np.array([T_bat_0])
        
    

        # Apply current state as constraint
        self.solver.set(0, "lbx", xt)
        self.solver.set(0, "ubx", xt)

        # Solve mpc and apply control
        self.solver.solve()
        self.total_cost += self.solver.get_cost()
        # ut = solver.get(0, "u").item()
        ut = self.solver.get(0, "u")
        slack_lower = self.solver.get(1, "sl")
        slack_upper = self.solver.get(1, "su")
        slx = slack_upper[2]
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
        print("total errors", self.total_errors)

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
            # Arbitrary
            #alpha_1 = 0.1 # 1 
            #alpha_2 = 2 # 0.980682  
            mdot_c = density_coolant*pump_displacement*omega
            # Dynamics
            
            # Get the cooler in and out temps
            NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
            T_clin= (self.T_bat_last + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
            T_clout = ((T_clin - self.T_bat_last) * alpha_2*np.exp(-NTU_bat) + self.T_bat_last)
          
            Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

            T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(self.T_env - self.T_bat_last))
                
            K1 = cs.DM.full(self.T_bat_dot_function(T_bat_k_minus_1, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale)).item()
            #K2 = self.T_bat_dot_function(T_bat_k_minus_1 + self.dt * K1, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale)
            #K3 = self.T_bat_dot_function(T_bat_k_minus_1 + self.dt * K2, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale)
            #K4 = self.T_bat_dot_function(T_bat_k_minus_1 +  K3, current, omega/self.omega_scale, Q_heat/self.Q_heat_scale)
            
            T_bat_pred = T_bat_k_minus_1 + (self.dt ) * (K1) # + 2*K2 + 2*K3 + K4)
            #T_bat_pred = T_bat_k_minus_1 + self.dt * (T_bat_dot_model)
            pred_error_nn = T_bat_pred - T_bat_k
            T_bat_pred_nn = T_bat_pred

        #u_N = self.solver.get(10,'u')
        #print('slacking off input', u_N)
        
        
        # Want to compare prediction at time step 0 with value at time step 1
        # So delay update of xt pred
        if self.current_iterate > 0:
            pred_error = self.xt_pred[0] - T_bat_0
        else:
            pred_error = 0
        T_bat_pred = self.xt_pred[0]
        pred = self.solver.get(1, 'x')
        
              
        #self.xt_pred = np.array([pred[0].item(), pred[1].item()])
        self.xt_pred = np.array([pred[0].item()])
        
        self.x_last = xt
        self.omega_last = omega_value
        self.Q_heat_last = Q_heat_value
        self.current_last = disturbances[int(self.dt*(self.current_iterate))].item()

        self.T_bat_last = T_bat_0
        if (self.current_iterate * self.dt) < self.T_warm_start:
            if T_bat_0 > T_bat_target: # Cooling regime
                omega_value, Q_heat_value = 4000*2*np.pi/60, -4000
            else: # Heating regime
                omega_value, Q_heat_value = 4000*2*np.pi/60, 4000
        if self.current_iterate > 0:
            self.collect_data(T_bat_0, dT_bat)
  
        if self.dt*self.current_iterate >= 2470:
            df = pd.DataFrame(data=self.residual_dictionary)
            df.to_csv("residuals.csv", index=False)

        elapsed = 1000*(time.time() - start)

        print(omega_value, Q_heat_value, self.xt_pred[0], T_bat_0, T_bat_target)


        print("cumulative cost", self.total_cost)
        print(elapsed, 'ms')
        print("--------------------------------")
        self.current_iterate += 1

        

        return omega_value, Q_heat_value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0