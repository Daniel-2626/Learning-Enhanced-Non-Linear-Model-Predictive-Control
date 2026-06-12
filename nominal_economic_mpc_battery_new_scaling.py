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

### Model of battery dynamics 
class BatteryDynamics:
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

        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        # Dynamics
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma*(T_env - T_bat))/model.T_bat_scale
        X_dot_nominal = cs.vertcat(T_bat_dot_model)
        X_dot_nominal = cs.vertcat(T_bat_dot_model)
        T_bat_dot_function = cs.Function("f_model", [T_bat_normalized, current_normalized, omega_normalized, Q_heat_normalized], 
        [T_bat_dot_model], ["T_bat_normalized", "current_normalized", "omega_normalized", "Q_heat_normalized"], ["ode"])

        f_expl = X_dot_nominal 
        x_start = np.array([T_env/model.T_bat_scale]) # initial constraint (gets overwritten)
        
        ## Power functions (for cost)
        Pump_power = (-0.21574 + 10.0501*omega_normalized**2 + 6.84987*omega_normalized**3)/1000 # Casadi pump fit 3rd degree, / 1000 to obtain in Kilowatts

        Heating_power = Q_heat/model.Q_heat_scale # Q_heat / Q_heat_scale = [W / 1000] = [kW]
        # y_expr will be used in cost function. Vector of [Pump power; Heating power; T_bat]
        model.cost_y_expr_0 = cs.vertcat(Pump_power, Heating_power, T_bat_normalized)
        model.cost_y_expr = cs.vertcat(Pump_power, Heating_power, T_bat_normalized)
        model.cost_y_expr_e = cs.vertcat(T_bat_normalized)
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
        model.name = "battery_nominal"
        model.T_bat_dot_function = T_bat_dot_function

        
        return model, constraint 
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
        ocp.model.cost_y_expr = model_ac.cost_y_expr
        ocp.model.cost_y_expr_e = model_ac.cost_y_expr_e
        ocp.model.cost_y_expr_0 = model_ac.cost_y_expr_0

        # Define weight matrices
        Q = np.array([15, 15, 0]) # note: no weighing on state
        Q = np.diag(Q)
        ocp.cost.W_0 = Q
        ocp.cost.W = Q
        ocp.cost.W_e = np.diag(np.array([0])) # Terminal cost only considers state, set this to 0 for no weighing

        ocp.cost.yref_0 = np.zeros((ny, ))
        ocp.cost.yref = np.zeros((ny, ))
        ocp.cost.yref_e = np.zeros((ny_e, ))

        # Initial state (will be overwritten)
        ocp.constraints.x0 = model.x_start 

        # Set constraints
        omega_max = (4000*2*np.pi/60)/model.omega_scale
        omega_min = (150*2*np.pi/60)/model.omega_scale

        Q_heat_max = 4000/model.Q_heat_scale
        Q_heat_min = -4000/model.Q_heat_scale
        Tb_max = 50 + CELSIUS_TO_KELVIN
        Tb_min = -20 + CELSIUS_TO_KELVIN
        SOC_max = 1
        SOC_min = 0
        
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


### Controller including setup, collect_data and get_input
class Controller:
    def setup(self, T_bat_target, T_env):

        # MPC Setup 
        self.N = 200 # Prediction horizon
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

        self.residual_dictionary = {'run': [], 'T_bat':[], 'current': [], 'omega_scaled': [], 'Q_heat_scaled': [], 'Q_cool':[], 'residual': [], 'T_bat_dot_model': [], 'T_bat_dot_euler': []}
        
        # Reading disturbance info
        df = pd.read_csv("current_rms_5s.csv", names=["current"])
        arr = df.to_numpy(dtype=np.float32)
        self.disturbance_values = arr  

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
        print("NOMINAL ECONOMIC MPC BATTERY MODEL")
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
            
            # Set disturbances (current)
            param_values = disturbances[self.current_iterate + k].item()/self.current_scale
            
            self.solver.set(k, "p", param_values)
        
        # Set terminal reference
        y_ref_terminal = np.array([T_bat_target/self.T_bat_scale]) # Terminal cost only on states
        self.solver.set(self.N, "yref", y_ref_terminal)
        # Set current on last time step
        param_values = disturbances[self.current_iterate + self.N].item()/self.current_scale

        self.solver.set(self.N, "p", param_values)

        start = time.time()

        xt = np.array([T_bat_0/self.T_bat_scale])

        # Apply current state as constraint
        self.solver.set(0, "lbx", xt)
        self.solver.set(0, "ubx", xt)

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
        
        # Want to compare prediction at time step 0 with value at time step 1
        # So delay update of xt pred
        if self.current_iterate > 0:
            pred_error = self.xt_pred[0] - T_bat_0
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
        # Save data to CSV
        if self.dt*self.current_iterate >= 3*2470:
            df = pd.DataFrame(data=self.residual_dictionary)
            df.to_csv("Offline_Training/Training_Data/matched_residuals_thermal_management_cooling.csv", index=False)
        # Collect data
        if self.current_iterate > 0:
            self.collect_data(T_bat_0, dT_bat)
  
        self.T_bat_last = T_bat_0
        if (self.current_iterate * self.dt) < self.T_warm_start or (status != 0):
            if T_bat_0 > T_bat_target: # Cooling regime
                omega_value, Q_heat_value = 4000*2*np.pi/60, -4000
            else: # Heating regime
                omega_value, Q_heat_value = 4000*2*np.pi/60, 4000



        elapsed = 1000*(time.time() - start)

        print(omega_value, Q_heat_value, self.T_bat_scale*self.xt_pred[0], T_bat_0, T_bat_target)


        print(elapsed, 'ms')
        print("--------------------------------")
        self.current_iterate += 1

        
        # Only omega_value and Q_heat_value is needed for control, rest for plotting (set to zero during last debugging)
        return omega_value, Q_heat_value, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, slack_x
