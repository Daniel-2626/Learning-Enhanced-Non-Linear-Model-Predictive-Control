import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd
class Model_Predictive_Controller:
    def MPC_setup(self, T_bat_target):

        ## CONFIG
        # MPC setup
        Q_T_bat = 10
        Q_SOC = 0 # Don't care what SOC is. Might cause problems in the future
        
        R_omega = 0.1
        R_Q_heat = 0.1
        t_horizon = 40
        N = 40 # number of look ahead steps
        step_horizon = t_horizon/N # time between steps in seconds

        # Constraints
        T_bat_max = 100 
        T_bat_min = -100 
        SOC_min = 0
        SOC_max = 1
        
        omega_max = 4000*2*np.pi/60
        omega_min = 0 

        Q_heat_max = 50 # W
        Q_heat_min = 0 # W
        # Target
        T_bat_target = T_bat_target
        SOC_target = 0 # No target actually
        # State symbolic variables
        T_bat = ca.SX.sym('T_bat')
        SOC = ca.SX.sym('SOC')
        states = ca.vertcat(
            T_bat,
            SOC
        )
        n_states = states.numel() # Returns amount of elements

        # control symbolic variables
        omega = ca.SX.sym('omega')
        Q_heat = ca.SX.sym('Q_heat')

        controls = ca.vertcat(
            omega,
            Q_heat
        )
        n_controls = controls.numel()

        # disturbances 
        current = ca.SX.sym("I_bat")
        disturbances = ca.vertcat(current)
        n_disturbances = disturbances.numel()

        # matrix containing all states over all time steps
        X = ca.SX.sym('X', n_states, N+1)

        # matrix containing all control actions over all time steps
        U = ca.SX.sym('U', n_controls, N)

        # column vector for storing initial state, target state AND disturbances 
        P = ca.SX.sym('P', n_states + n_states + n_disturbances)

        # State weights matrix
        Q = ca.diagcat(Q_T_bat, Q_SOC)

        # Controls weight matrix
        R = ca.diagcat(R_omega, R_Q_heat)

        # discretization model (e.g. x2 = f(x1, v, t) = x1 + v*dt)
        # Parameters
        m_battery = 20*2.5*5
        c_battery = 795
        c_coolant = 3500
        density_coolant = 1050
        pump_displacement = 1/(2*np.pi)*40/(100^3) # D parameter in simulink
        R_battery = 4*20*0.0128 # Battery resistance
        C_battery = 28*3600 # in coloumb
        K_heater = 500
        alpha = 0.65
        ## Controller model 
        T_bat_dot_model = alpha/(m_battery*c_battery) * (current**2 * R_battery + density_coolant*pump_displacement*omega*c_coolant*Q_heat/K_heater)
        SOC_dot_model = -current/C_battery
        ode_model = ca.vertcat(T_bat_dot_model, SOC_dot_model)

        f_model = ca.Function("f_model", [states,controls,disturbances], [ode_model], ["x", "u", "d"], ["ode"])
        
        cost_fn = 0 # Cost function
        g = X[:, 0] - P[:n_states] # constraints in the equation, that x0 = p0 

        # Runge Kutta
        for k in range(N):
            # model update, what our controller knows
            st = X[:, k]
            contr = U[:, k]
            disturbances = P[-n_disturbances:]

            cost_fn = (cost_fn +
                    (st-P[n_states:2*n_states]).T @ Q @ (st-P[n_states:2*n_states]) +
                    contr.T @ R @ contr
                    )
            st_next = X[:, k+1]
            K1 = f_model(st, contr, disturbances)
            K2 = f_model(st + step_horizon/2 * K1, contr, disturbances)
            K3 = f_model(st + step_horizon/2 * K2, contr, disturbances)
            K4 = f_model(st + step_horizon * K3, contr, disturbances)
            st_next_RK4 = st + (step_horizon / 6) * (K1 + 2*K2 + 2*K3 + K4)
            g = ca.vertcat(g, st_next - st_next_RK4) # Basically saying X[:, k+1] = runge kutta evaluation
        cost_fn = cost_fn +  (X[:, N]-P[n_states:2*n_states]).T @ Q @ (X[:, N]-P[n_states:2*n_states])
        OPT_variables = ca.vertcat(
            X.reshape((-1,1)),    # E.g. 3 x 11 --> 33 x 1
            U.reshape((-1,1))     # E.g. 4 x 10 --> 40 x 1
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

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
        ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1)) # long column vector 

        lbx[0: n_states*(N+1): n_states] = T_bat_min # Different notation to matlab. Here have [start:stop.step]. We fill out the vector related to states [start:n_states*(N+1)] with a lower bound every n_states, meaning we set lower bound for first state at each time step
        lbx[1: n_states*(N+1): n_states] = SOC_min # Here set lower bound for second state

        ubx[0: n_states*(N+1): n_states] = T_bat_max 
        ubx[1: n_states*(N+1): n_states] = SOC_max 

        lbx[n_states*(N+1): -1: n_controls] = omega_min 
        ubx[n_states*(N+1): -1: n_controls] = omega_max 
        
        lbx[1+ n_states*(N+1): -1: n_controls] = Q_heat_min 
        ubx[1+ n_states*(N+1): -1: n_controls] = Q_heat_max 

        
        

        args = {
            'lbg': ca.DM.zeros((n_states*(N+1), 1)), # constrainst lower bound (basically giving == constraint)
            'ubg': ca.DM.zeros((n_states*(N+1), 1)),
            'lbx': lbx, 
            'ubx': ubx
        }

        self.solver = solver
        self.args = args
        self.state_target = ca.DM([T_bat_target, SOC_target])

        self.u0 = ca.DM.zeros((n_controls, N))
        self.X0 = ca.DM.zeros((n_states, N+1))

        self.n_states = n_states
        self.n_controls = n_controls
        self.N = N

    def DM2Arr(self,dm):
    # returns a full matrix instead if a soarse ibe
        return np.array(dm.full())
    def MPC_step(self, T_bat_0, SOC_0, current):
        state_init = ca.DM([T_bat_0, SOC_0])
        state_target = self.state_target
        X0 = self.X0
        u0 = self.u0
        args = self.args
        solver = self.solver
        n_states = self.n_states
        n_controls = self.n_controls
        N = self.N
        args['p'] = ca.vertcat(
            state_init, # current state
            state_target,
            current 
        )

        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )
        # sol.x0 is warm starting it
        sol = solver(
            x0 = args['x0'],
            lbx = args['lbx'],
            ubx = args['ubx'],
            lbg = args['lbg'],
            ubg = args['ubg'],
            p = args['p']
        )
        
        self.u0 = ca.reshape(sol['x'][n_states*(N+1):], n_controls, N   ) # gives u as a vector where rows are the different inputs and columns the time steps (applied input at col 0)
        self.X0 = ca.reshape(sol['x'][:n_states*(N+1)], n_states, N+1)

        inp = self.DM2Arr(self.u0[:, 0])
        omega = inp[0].item()
        Q_heat = inp[1].item()
        return omega, Q_heat
        