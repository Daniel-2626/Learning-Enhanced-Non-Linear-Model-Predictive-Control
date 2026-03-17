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
    def __init__(self, T_env): # remove gym_env for cascaded tank
        self.T_env = T_env
    def model(self): # remove gym_env for cascaded tank
        model = cs.types.SimpleNamespace()
        model.omega_scale = 100
        model.Q_heat_scale = 1000
        T_bat = cs.MX.sym('T_bat')
        SOC= cs.MX.sym('SOC')
        X = cs.vertcat(T_bat, SOC)

        omega_normalized = cs.MX.sym('omega_norm')
        omega = model.omega_scale * omega_normalized #cs.MX.sym('omega') # Power into heater/cooler given as efficiency*Pin
        Q_heat_normalized = cs.MX.sym('Q_heat_norm')
        Q_heat = model.Q_heat_scale * Q_heat_normalized #cs.MX.sym('Q_heat') # Pump control (rpm that is converted to kg/s)
        current = cs.MX.sym('current')
        
        nn_on = cs.MX.sym('nn_on') # Flip switch for whether on not to have the NN in the model (helps when NN not trained yet)
        
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

        T_env = self.T_env

        #[0.637968, 1, 0.980682, 0.981036]
        alpha_0 = 0.637968 #0.65
        alpha_1 = 1 # 0.99 #0.998427 #0.99
        alpha_2 = 0.980682  #0.999686 #.97
        alpha_3 = 0.981036
        gamma = 7.59853
        
        mdot_c = density_coolant*pump_displacement*omega
        # Dynamics
        
        # Get the cooler in and out temps
        NTU_bat  = (alpha_3*hA_bat) / (mdot_c*c_coolant + 1e-3)
        T_clin= (T_bat + alpha_1*(1/(1-np.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
        T_clout = ((T_clin - T_bat) * alpha_2*np.exp(-NTU_bat) + T_bat)
        
        constraint = cs.types.SimpleNamespace()
        # Errors if set to -10, look into better formulation with slack variables
        constraint.T_clin_min = -10 + CELSIUS_TO_KELVIN
        constraint.T_clin_max = 100 + CELSIUS_TO_KELVIN
        constraint.T_clout_min = -10 + CELSIUS_TO_KELVIN
        constraint.T_clout_max = 100 + CELSIUS_TO_KELVIN
        constraint.expr = cs.vertcat(T_clin, T_clout)

        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        # Now for the actual calculations 
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))
        SOC_dot_model = -current/C_battery
        X_dot_nominal = cs.vertcat(T_bat_dot_model, SOC_dot_model)

        f_expl = X_dot_nominal 
        x_start = np.array([-10+CELSIUS_TO_KELVIN,1]) # initial constraint (gets overwritten)

        # store to struct
        
        model.x = X 
        model.xdot = cs.MX.sym('xdot', 2)
        model.u = U
        model.z = cs.vertcat([])
        model.p = current
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
        #l4c_y_expr = None

        # Define weight parameters
        Q = np.diag([10, 0.1])
        R = np.diag([1, 0.1])
        ocp.cost.W = scipy.linalg.block_diag(Q,R)
        print(ocp.cost.W)
        # Initial state (will be overwritten?)
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
        Q_heat_min = 0/model.Q_heat_scale
        Tb_max = 50 + CELSIUS_TO_KELVIN
        Tb_min = -10 + CELSIUS_TO_KELVIN
        SOC_max = 1
        SOC_min = 0
        ocp.constraints.lbu = np.array([omega_min, Q_heat_min])
        ocp.constraints.ubu = np.array([omega_max, Q_heat_max])

        ocp.constraints.idxbu = np.array([0,1])
        ocp.constraints.idxbx = np.array([0]) # at what indices to have constraints
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
        ocp.cost.zl = 1000 * np.zeros((ns,))
        ocp.cost.zu = 1000 * np.zeros((ns,))
        ocp.cost.Zl = 1000* np.ones((ns,))
        ocp.cost.Zu = 1000 * np.ones((ns,))

        ocp.cost.zl_0 = 1000 * np.zeros((nsh+nsu,))
        ocp.cost.zu_0 = 1000 * np.zeros((nsh+nsu,))
        ocp.cost.Zl_0 = 1000* np.ones((nsh+nsu,))
        ocp.cost.Zu_0 = 1000 * np.ones((nsh+nsu,))

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
        print(model_ac.u)
        model_ac.p = model.p
        model_ac.name = model.name
        model_ac.con_h_expr = constraint.expr
        return model_ac



class Controller:
    def setup(self, T_bat_target, T_env):

        # MPC Setup 
        self.N = 40
        t_horizon = 40*5
        model = BatteryDynamics(T_env=T_env)
      
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
        self.omega_scale = casadi_model.omega_scale
        self.Q_heat_scale = casadi_model.Q_heat_scale
        # Reading disturbance info
        df = pd.read_csv("current_intp1.csv", names=["current"])
        arr = df.to_numpy(dtype=np.float32)
        self.disturbance_values = arr
        

    def get_input(self, T_bat_target, T_bat_0, SOC_0, current_0, dT_bat, dSOC):
       
        print("--------------------------------")
        print("NOMINAL ACADOS MPC BATTERY MODEL SCALED")
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
            print(self.dt*(self.current_iterate + k))
            param_values = disturbances[int(self.dt*(self.current_iterate + k))].item()
            self.solver.set(k, "p", param_values)

        # Set terminal reference
        # y_ref_terminal = np.array([Tb_ref, SOC_ref])
        y_ref_terminal = np.array([T_bat_target, 0.0]) # Terminal cost only on states so 2x1 instead of 4x1 above
        #print(y_ref_terminal)
        self.solver.set(self.N, "yref", y_ref_terminal)
        param_values = disturbances[int(self.dt*(self.current_iterate + self.N))].item()

        self.solver.set(self.N, "p", param_values)


        start = time.time()
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
        
        omega_value = self.omega_scale * ut[0].item()
        Q_heat_value = self.Q_heat_scale * ut[1].item()
        
        ut_p1 = self.solver.get(1, "u")
        print("inputs normalized", ut_p1)
        status = self.solver.get_status()
        if status != 0:
            print("ERROR", status, "at iterate", self.current_iterate)
            self.total_errors += 1
        #u_N = self.solver.get(10,'u')
        #print('slacking off input', u_N)
        
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
        return omega_value, Q_heat_value, self.xt_pred[0]