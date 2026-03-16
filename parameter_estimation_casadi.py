import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from time import time
import pandas as pd
CELSIUS_TO_KELVIN = 273.15
class Estimator:
    def estimator_setup(self):

        ## CONFIG
        # MPC setup
        Q_T_bat = 1        
        
        self.T_estim_start = 0
        N = 2400 # number of look ahead steps
        t_horizon = N
        step_horizon = t_horizon/N # time between steps in seconds

        # Constraints
        T_bat_max = 50 + CELSIUS_TO_KELVIN
        T_bat_min = -20 + CELSIUS_TO_KELVIN
        
        # State symbolic variables
        T_bat = ca.SX.sym('T_bat')
        
        states = ca.vertcat(
            T_bat,
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
        P = ca.SX.sym('P', n_states + n_states + (n_states + n_disturbances+n_controls)*N)

        # State weights matrix
        Q = ca.diagcat(Q_T_bat)


        # discretization model (e.g. x2 = f(x1, v, t) = x1 + v*dt)
        # Parameters
        m_battery = 20*2.5*4
        c_battery = 795
        c_coolant = 3500
        density_coolant = 1050
        pump_displacement = 1/(2*np.pi)*40/(100**3) # D parameter in simulink
        mdot_c = density_coolant*pump_displacement*omega
        R_battery = 4*20*0.0128 # Battery resistance
        C_battery = 28*3600 # in coloumb
        hA_bat = 2500
        T_env = 0 + CELSIUS_TO_KELVIN
        alpha_0 = ca.SX.sym("alpha_0")
        
        alpha_1 = ca.SX.sym("alpha_1")
        alpha_2 = ca.SX.sym("alpha_2")
        alpha_3 = ca.SX.sym("alpha_3")
        alpha_min = 0.5 #0.8
        alpha_max = 1 #1.2
        alphas = ca.vertcat(alpha_0, alpha_1, alpha_2, alpha_3)
        n_alphas = alphas.numel()

        gamma = ca.SX.sym("gamma")
        gammas = ca.vertcat(gamma)
        gamma_min = 0 # Should be positive with current formulation
        gamma_max = 100
        n_gammas = gammas.numel()        
        # Get the cooler in and out temps
        NTU_bat  = (alpha_3 * hA_bat) / (mdot_c*c_coolant + 1e-3)
        T_clin= (T_bat + alpha_1*(1/(1-ca.exp(-NTU_bat)))*Q_heat/(mdot_c*c_coolant + 1e-3))
        T_clout = ((T_clin - T_bat) * alpha_2*ca.exp(-NTU_bat) + T_bat)
        T_clin_min = CELSIUS_TO_KELVIN 
        T_clin_max = 50 + CELSIUS_TO_KELVIN
        T_clout_min = CELSIUS_TO_KELVIN 
        T_clout_max = 50 + CELSIUS_TO_KELVIN
        Q_cool = mdot_c*c_coolant*(T_clout - T_clin)

        # Now for the actual calculations 
        T_bat_dot_model = alpha_0/(m_battery*c_battery) * (current**2 * R_battery - Q_cool + gamma * (T_env - T_bat))
        ode_model = T_bat_dot_model

        # ## Controller model 
        # T_bat_dot_model = alpha/(m_battery*c_battery) * (current**2 * R_battery + density_coolant*pump_displacement*omega*c_coolant*Q_heat/K_heater)
        # SOC_dot_model = -current/C_battery
        # ode_model = ca.vertcat(T_bat_dot_model, SOC_dot_model)

        f_model = ca.Function("f_model", [states,controls,disturbances, alphas, gammas], [ode_model], ["x", "u", "d", "alpha", "gammas"], ["ode"])
        T_clin_fun = ca.Function("T_clin_fun", [states,controls, alphas, gammas], [T_clin], ["x", "u", "alpha", "gammas"], ["T_clin_fun"])
        T_clout_fun = ca.Function("T_clout_fun", [states,controls, alphas, gammas], [T_clout], ["x", "u", "alpha", "gammas"], ["T_clout_fun"])

        cost_fn = 0 # Cost function
        g = X[:, 0] - P[:n_states] # constraints in the equation, that x0 = p0 
        weight = 1
        # Runge Kutta
        for k in range(N):
            # model update, what our controller knows
            st = X[:, k]
            if k > 1000:
                weight = 1
            contr = np.array([P[2*n_states + N + k], P[2*n_states + 2*N + k]])
            disturbances = P[ -N+k]

            cost_fn = (cost_fn +
                    weight*(st-P[n_states+k]).T @ Q @ (st-P[n_states+k])
                    )
            T_clin_eval = T_clin_fun(st, contr, alphas, gammas)
            T_clout_eval = T_clout_fun(st, contr, alphas, gammas)

            st_next = X[:, k+1]
            K1 = f_model(st, contr, disturbances, alphas, gammas)
            #K2 = f_model(st + step_horizon/2 * K1, contr, disturbances, alphas)
            #K3 = f_model(st + step_horizon/2 * K2, contr, disturbances, alphas)
            #K4 = f_model(st + step_horizon * K3, contr, disturbances, alphas)
            st_next_RK4 = st + (step_horizon) * (K1) #  + 2*K2 + 2*K3 + K4)
            g = ca.vertcat(g, T_clin_eval, T_clout_eval, st_next - st_next_RK4) # Basically saying X[:, k+1] = runge kutta evaluation
        g = ca.vertcat(g, T_clin_eval, T_clout_eval)
        cost_fn = cost_fn +  weight*(X[:, N]-P[n_states + N]).T @ Q @ (X[:, N]-P[n_states + N])
        OPT_variables = ca.vertcat(
            X.reshape((-1,1)),    # E.g. 3 x 11 --> 33 x 1
            alphas.reshape((-1,1)),     # E.g. 4 x 10 --> 40 x 1
            gammas.reshape((-1,1))
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

        lbx = ca.DM.zeros((n_states*(N+1) + n_alphas + n_gammas, 1))
        ubx = ca.DM.zeros((n_states*(N+1) + n_alphas + n_gammas, 1)) # long column vector 

        lbx[0: n_states*(N+1)] = T_bat_min # Different notation to matlab. Here have [start:stop.step]. We fill out the vector related to states [start:n_states*(N+1)] with a lower bound every n_states, meaning we set lower bound for first state at each time step
        lbx[-n_alphas-n_gammas:-n_gammas] = alpha_min
        lbx[-n_gammas:] = gamma_min

        ubx[0: n_states*(N+1)] = T_bat_max 
        ubx[-n_alphas-n_gammas:-n_gammas] = alpha_max
        ubx[-n_gammas:] = gamma_max

        lg = ca.DM.zeros(3*(n_states*(N+1)), 1)
        ug = ca.DM.zeros(3*(n_states)*(N+1), 1)

        lg[1::3] = T_clin_min # Odd is T_clin
        ug[1::3] = T_clin_max
        lg[2::3] = T_clout_min # Odd is T_clin
        ug[2::3] = T_clout_max
        args = {
            'lbg': lg, # constrainst lower bound (basically giving == constraint)
            'ubg':  ug,
            'lbx': lbx, 
            'ubx': ubx
        }

        self.solver = solver
        self.args = args


        self.X0 = ca.DM.ones((n_states * (N+1) + n_alphas + n_gammas, 1))

        self.n_states = n_states
        self.n_alphas = n_alphas
        self.n_gammas = n_gammas
        self.N = N

    def DM2Arr(self,dm):
    # returns a full matrix instead if a sparse one
        return np.array(dm.full())
    def get_estimate(self):
        
        df = pd.read_csv("true_data_intp1.csv", names=["T_bat", "input_omega", "input_q_heat", "current"])
        arr = df.to_numpy(dtype=np.float32)
        #print(df.head())
        T_estim_start = self.T_estim_start
        N = self.N
        n_states = self.n_states
        T_bat_true = np.array(arr[T_estim_start:T_estim_start + N+2,0])
        T_bat_true = np.reshape(T_bat_true, (N+2,1))
        omega_true = np.array(arr[T_estim_start:T_estim_start + N,1])
        #print(omega_true)
        omega_true = np.reshape(omega_true, (N,1))
        Q_heat_true = np.array(arr[T_estim_start:T_estim_start + N,2])
        Q_heat_true = np.reshape(Q_heat_true, (N,1))
        current = np.array(arr[T_estim_start:T_estim_start + N,3])
        current = np.reshape(current, (N,1))
        T_bat_0 = T_bat_true[0]
        #print(T_bat_0)
        start = time()
        
        
        
        X0 = self.X0
        args = self.args
        solver = self.solver
        n_states = self.n_states
        n_alphas = self.n_alphas
        N = self.N
        args['p'] = ca.vertcat(
            T_bat_true,
            omega_true,
            Q_heat_true,
            current 
        )

        # optimization variable current state
        args['x0'] = X0
        # sol.x0 is warm starting it
        sol = solver(
            x0 = args['x0'],
            lbx = args['lbx'],
            ubx = args['ubx'],
            lbg = args['lbg'],
            ubg = args['ubg'],
            p = args['p']
        )
        
        sol_x = sol['x'][:self.N+1]
        alpha_fit = sol['x'][-self.n_alphas-self.n_gammas:-self.n_gammas]
        gamma_fit = sol['x'][-self.n_gammas:]
        print(sol_x)
        print(alpha_fit)
        print(gamma_fit)

        end = time()
        print(end - start, "s")

        return sol_x, alpha_fit, gamma_fit
        
myEstimator = Estimator()
myEstimator.estimator_setup()
sol_x, alpha_fit, gamma_fit = myEstimator.get_estimate()
df = pd.read_csv("true_data_intp1.csv", names=["T_bat", "input_omega", "input_q_heat", "current"])
arr = df.to_numpy(dtype=np.float32)
T_estim_start = myEstimator.T_estim_start
N = myEstimator.N
n_states = myEstimator.n_states
T_bat_true = np.array(arr[T_estim_start:T_estim_start + N+2,0])
plt.plot(sol_x)
plt.plot(T_bat_true)
plt.legend(["Model", "True"])
plt.show()

# IF ALPHA MIN=0.1, ALPHA MAX = 10 the optimizer sucks
# IF Alpha min = 0.9 alpha max = 1.1 it is great 
# 