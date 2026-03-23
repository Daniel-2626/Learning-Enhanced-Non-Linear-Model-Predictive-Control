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
        pass #
    def model(self): # remove gym_env for cascaded tank

        T_bat = cs.MX.sym('T_bat')
        SOC= cs.MX.sym('SOC')
        X = cs.vertcat(T_bat, SOC)
        omega = cs.MX.sym('omega') # Power into heater/cooler given as efficiency*Pin
        Q_heat = cs.MX.sym('Q_heat') # Pump control (rpm that is converted to kg/s)
        current = cs.MX.sym('current')
        theta_1 = cs.MX.sym("theta_1")
        theta_2 = cs.MX.sym("theta_2")
        U = cs.vertcat(omega, Q_heat)
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
        
        alpha_0 = 0.65
        alpha_1 = 0.99
        alpha_2 = 0.97
        kappa = 1e-4
        mdot_c = density_coolant*pump_displacement*omega
        


        # Dynamics
        
        # Get the cooler in and out temps
        NTU_bat  = (hA_bat) / (mdot_c*c_coolant + 1e-3)
        T_clin= alpha_1*(T_bat + 1/(1-np.exp(-NTU_bat))*Q_heat/(mdot_c*c_coolant + 1e-3))
        T_clout = alpha_2*((T_clin - T_bat) * np.exp(-NTU_bat) + T_bat)
        
        constraint = cs.types.SimpleNamespace()
        constraint.T_clin_min = -20 + CELSIUS_TO_KELVIN
        constraint.T_clin_max = 100 + CELSIUS_TO_KELVIN
        constraint.T_clout_min = -20 + CELSIUS_TO_KELVIN
        constraint.T_clout_max = 100 + CELSIUS_TO_KELVIN
        constraint.expr = cs.vertcat(T_clin, T_clout)

        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        # Now for the actual calculations 
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * ((R_battery+theta_1) * current**2  - (1 + theta_2)*Q_cool)
        SOC_dot_model = -current/C_battery
        X_dot_nominal = cs.vertcat(T_bat_dot_model, SOC_dot_model)

        f_expl = X_dot_nominal 
        x_start = np.array([-10+CELSIUS_TO_KELVIN,1]) # initial constraint (gets overwritten)

        # store to struct
        model = cs.types.SimpleNamespace()
        model.x = X 
        model.xdot = cs.MX.sym('xdot', 2)
        model.u = U
        model.z = cs.vertcat([])
        model.p = cs.vertcat(current, theta_1, theta_2)
        model.f_expl = f_expl
        model.f_nominal = X_dot_nominal
        model.x_start = x_start
        model.constraints = cs.vertcat([]) # add constraints here or in mpc?
        model.name = "battery_nominal"
        
        return model, constraint 

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
        nx = 2
        nu = 2
        ny = nx + nu # Stage cost
        ny_e = nx # Terminal cost considers only states
        nh = constraint.expr.shape[0]
        
        nsh = nh
        ns = nsh

        # Create ocp to formulate optim
        ocp = AcadosOcp()
        ocp.model = model_ac
        ocp.dims.N = N
        ocp.dims.nx = nx
        ocp.dims.nu = nu
        ocp.dims.ny = ny
        ocp.dims.nh = nh

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
        #l4c_y_expr = None

        # Define weight parameters
        Q = np.diag([10000, 0.1])
        R = np.diag([0.0001, 0.0000001])
        ocp.cost.W = scipy.linalg.block_diag(Q,R)

        # Initial state (will be overwritten?)
        ocp.cost.W_e = Q 
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # Initial state
        ocp.constraints.x0 = model.x_start

        # Set constraints
        omega_max = 4000*2*np.pi/60
        
        omega_min = 150*2*np.pi/60
        print(omega_min)
        Q_heat_max = 4000                
     
     
        Q_heat_min = 0
        Tb_max = 50 + CELSIUS_TO_KELVIN
        Tb_min = -20 + CELSIUS_TO_KELVIN
        SOC_max = 1
        SOC_min = 0
        ocp.constraints.lbu = np.array([omega_min, Q_heat_min])
        ocp.constraints.ubu = np.array([omega_max, Q_heat_max])
        ocp.constraints.idxbu = np.array([0,1])
        ocp.constraints.idxbx = np.array([0]) # at what indices to have constraints? 
        ocp.constraints.ubx = np.array([Tb_max])
        ocp.constraints.lbx = np.array([Tb_min])

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
        ocp.cost.zl = 10000000 * np.ones((ns,))
        ocp.cost.zu = 10000000 * np.ones((ns,))
        ocp.cost.Zl = 10000000* np.ones((ns,))
        ocp.cost.Zu = 10000000 * np.ones((ns,))

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

        ocp.constraints.lsh = np.zeros(nsh)
        ocp.constraints.ush = np.zeros(nsh)
        ocp.constraints.idxsh = np.array(range(nsh))
        
        # Solver options
        ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.nlp_solver_type = "SQP_RTI"
        
        # Will be overwritten
        ocp.parameter_values = np.zeros((3,1))

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
    def setup(self, T_bat_target):

        # MPC Setup 
        self.N = 40
        t_horizon = 40
        model = BatteryDynamics()
      
        casadi_model, constraint = model.model()
        
        # MIGHT CAUSE ISSUES SINCE DIFFERENT THAN YU MEI, here model=casadi_model
        self.solver = MPC(model=casadi_model, constraint=constraint, N=self.N, t_horizon = t_horizon).solver # Returns the solver object from MPC

         
        

        # SOC_ref = None # This is not actually tracked
        self.dt = t_horizon/self.N
        self.obs_buffer = []
        self.batch_size = self.N # at most self.N - self.T_warm_start at least
        self.T_update = 60 
        self.T_warm_start = 10
        self.current_iterate = 0
        self.xt_pred = np.array([CELSIUS_TO_KELVIN, 1])
        self.x_last = np.array([0,0])
        self.omega_last = 0
        self.Q_heat_last = 0
        self.current_last = 0
        self.data = []
        self.total_errors = 0
        self.total_cost = 0
        self.theta = np.array([[0],[0]])
        # Reading disturbance info
        df = pd.read_csv("current_intp1.csv", names=["current"])
        arr = df.to_numpy(dtype=np.float32)
        self.disturbance_values = arr
        

    def get_input(self, T_bat_target, T_bat_0, SOC_0, current_0, dT_bat, dSOC):
       
        print("--------------------------------")
        print("LMS ACADOS MPC BATTERY MODEL WITH SLACK")
        if self.current_iterate == 0:
            self.xt_pred = ([T_bat_0, SOC_0])
        disturbances = self.disturbance_values

        # Set reference for each step in MPC horizon
        #print(disturbances[0], current_0)
        for k in range(self.N):
            #if k == 0: # only store first reference
                #Tb_ref_history.append(Tb_ref)
                # SOC_ref_history.append(SOC_ref)
            # Set terminal reference
            # y_ref_k = np.array([Tb_ref, SOC_ref, 0])
            y_ref_k = np.array([T_bat_target, 0.0, 0.0, 0.0])
            self.solver.set(k, "yref", y_ref_k)
            param_values = np.concatenate([[disturbances[k]], self.theta])
            
            self.solver.set(k, "p", param_values)

        # Set terminal reference
        # y_ref_terminal = np.array([Tb_ref, SOC_ref])
        y_ref_terminal = np.array([T_bat_target, 0.0]) # Terminal cost only on states so 2x1 instead of 4x1 above
        #print(y_ref_terminal)
        self.solver.set(self.N, "yref", y_ref_terminal)
        param_values = np.concatenate([[disturbances[self.N]], self.theta])

        self.solver.set(self.N, "p", param_values)


        start = time.time()
        xt = np.array([T_bat_0,SOC_0])
        # Apply current state as constraint
        self.solver.set(0, "lbx", xt)
        self.solver.set(0, "ubx", xt)

        # Solve mpc and apply control
        self.solver.solve()
        self.total_cost += self.solver.get_cost()/(1e8)
        # ut = solver.get(0, "u").item()
        ut = self.solver.get(0, "u")
        slack_lower = self.solver.get(1, "sl")
        slack_upper = self.solver.get(1, "su")
        print(slack_lower, slack_upper)
        omega_value = ut[0].item()
        Q_heat_value = ut[1].item()
        status = self.solver.get_status()
        
        
        if status != 0:
            print("ERROR", status, "at iterate", self.current_iterate)
            self.total_errors += 1
        #u_N = self.solver.get(10,'u')
        #print('slacking off input', u_N)
        Q_cool_eval = 0
        # LMS update
        if self.current_iterate > self.T_warm_start + 1:
            T_bat_pred = self.xt_pred[0].item()
            T_bat_last = self.x_last[0].item()
            omega_last = self.omega_last
            Q_heat_last = self.Q_heat_last
            current_last = disturbances[self.current_iterate-1].item()
            pred_error = np.array([[T_bat_0 - T_bat_pred]])
            T_bat = cs.MX.sym('T_bat')
            
            omega = cs.MX.sym('omega') # Power into heater/cooler given as efficiency*Pin
            Q_heat = cs.MX.sym('Q_heat') # Pump control (rpm that is converted to kg/s)
            current = cs.MX.sym('current')
            
            # Parameters
            m_battery = 20*2.5*4
            c_battery = 795
            c_coolant = 3500
            density_coolant = 1050
            pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
            hA_bat = 2500
                
            alpha_0 = 0.65
            alpha_1 = 0.99
            alpha_2 = 0.97
            kappa = 1e-4
            mdot_c = density_coolant*pump_displacement*omega

            # Dynamics
            
            # Get the cooler in and out temps
            NTU_bat  = (hA_bat) / (mdot_c*c_coolant + 1e-3)
            T_clin= alpha_1*(T_bat + 1/(1-np.exp(-NTU_bat))*Q_heat/(mdot_c*c_coolant + 1e-3))
            T_clout = alpha_2*((T_clin - T_bat) * np.exp(-NTU_bat) + T_bat)
            
            Q_cool = mdot_c*c_coolant*(T_clout - T_clin)
            Q_cool_f = cs.Function("Q_cool_f", [T_bat, omega, Q_heat], [Q_cool], ["T_bat", "omega", "Q_heat"], ["Q_cool"])

            Q_cool_eval = np.array(Q_cool_f(T_bat_last, omega_last, Q_heat_last).full())
            #print(Q_cool_eval)
            
            print("pred error", pred_error)
            current_sq = np.array([[0]]) #np.array([[current_last**2]])
            G = np.concatenate([current_sq, -alpha_0/(m_battery*c_battery) * Q_cool_eval])
            mu = 1e-6
            self.theta = self.theta + G@pred_error

            print("theta", self.theta)   
            Q_cool_eval = Q_cool_eval.item()
            
        # Want to compare prediction at time step 0 with value at time step 1
        # So delay update of xt pred
        pred = self.solver.get(1, 'x')
        
        self.xt_pred = np.array([pred[0].item(), pred[1].item()])
        
        self.x_last = xt
        
        self.omega_last = omega_value
        
        self.Q_heat_last = Q_heat_value
        self.current_last = current_0

 
        elapsed = time.time() - start
        
        print("total errors", self.total_errors)
    
        self.current_iterate += 1
        if self.current_iterate < self.T_warm_start:
            omega_value, Q_heat_value = 4000*2*np.pi/60, 4000
        print(omega_value, Q_heat_value, self.xt_pred[0], T_bat_0, T_bat_target)
        print("cumulative cost", self.total_cost)
        print(elapsed, 'ms')
        print("--------------------------------")
        return omega_value, Q_heat_value, self.xt_pred[0], Q_cool_eval