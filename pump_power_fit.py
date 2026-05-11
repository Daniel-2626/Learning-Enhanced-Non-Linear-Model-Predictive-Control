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
        Q_P = 100        
        
        self.T_estim_start = 0
        N = 54
        # Constraints
        P_max = ca.inf
        P_min = 0#-ca.inf
        
        # State symbolic variables
        Power = ca.SX.sym('Power')
        
        states = ca.vertcat(
            Power,
        )
        n_states = states.numel() # Returns amount of elements

        # control symbolic variables
        omega = ca.SX.sym('omega')

        controls = ca.vertcat(
            omega
        )
        n_controls = controls.numel()


        # matrix containing all states over all time steps
        X = ca.SX.sym('X', n_states, N)

        # matrix containing all control actions over all time steps
        U = ca.SX.sym('U', n_controls, N)

        # column vector for storing initial states, and states and controls for all shooting nodes (will be fed to the optimizer)
        P = ca.SX.sym('P', (n_states +n_controls)*N)

        # State weights matrix
        Q = ca.diagcat(Q_P)


        # Use a 2nd degree polynomial for pump power
        
        alpha_0 = ca.SX.sym("alpha_0")
        alpha_1 = ca.SX.sym("alpha_1")
        alpha_2 = ca.SX.sym("alpha_2")
        alpha_3 = ca.SX.sym("alpha_3")

        alpha_min = 0 #0 #np.array([-1000, -1000, -1000]) #0.8
        #alpha_min = alpha_min.reshape((-1,1))
        alpha_max = 1000 # np.array([1000,1000,1000]) #1.2
        #alpha_max = alpha_max.reshape((-1,1))
        alphas = ca.vertcat(alpha_0, alpha_1, alpha_2, alpha_3)
        n_alphas = alphas.numel()
        print("n_alphas", n_alphas)
        # Now for the actual calculations 
        Power_model = alpha_0*1 + alpha_1*omega + alpha_2*omega**2 + alpha_3*omega**3


        f_model = ca.Function("f_model", [controls, alphas], [Power_model], ["u", "alpha"], ["power"])

        cost_fn = 0 # Cost function
        weight = 1
        
        contr = P[N*n_states]


        #g = ca.vertcat(st)# st # constraints in the equation, that x0 = p0 
        st = P[0] #X[:,0]
        g = st - P[:n_states] # constraints in the equation, that x0 = p0 
        
        cost_fn = (weight*(st-P[0]).T @ Q @ (st-P[0]))

        # Runge Kutta
        for k in range(1,N):
            # model update, what our controller knows
            print(k)
            contr = P[N*n_states + k]

            st = P[k]

            st_fun = f_model(contr, alphas) 

            print(st)
            cost_fn = (cost_fn +
                    weight*(st-st_fun).T @ Q @ (st-st_fun)
                    )

            
            g = ca.vertcat(g, st_fun) # Basically saying X[:, k+1] = runge kutta evaluation
        #g = ca.vertcat(g, T_clin_eval, T_clout_eval)
        print(g)
        #cost_fn = cost_fn +  weight*(X[:, N]-P[n_states + N]).T @ Q @ (X[:, N]-P[n_states + N])
        OPT_variables = ca.vertcat(
            alphas.reshape((-1,1))
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
                'print_level': 5,
                'acceptable_tol': 1e-8,
                'acceptable_obj_change_tol': 1e-6
            },
            'print_time': 0
        }

        solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)

        lbx = ca.DM.zeros((n_alphas, 1))
        ubx = ca.DM.zeros((n_alphas, 1)) # long column vector 
        #print(lbx[0:n_alphas].shape)
        #print(alpha_min.shape)
        #lbx[:N*n_states] = P_min
        #ubx[:N*n_states] = P_max
        
       # Different notation to matlab. Here have [start:stop.step]. We fill out the vector related to states [start:n_states*(N+1)] with a lower bound every n_states, meaning we set lower bound for first state at each time step
        #lbx = alpha_min
        lbx[0] = -ca.inf
        lbx[1:] = alpha_min 
        ubx = alpha_max 

        lg = ca.DM.zeros((n_states*(N)), 1)
        ug = ca.DM.zeros((n_states)*(N), 1)

        lg = P_min # Odd is T_clin
        ug = P_max
        
        args = {
            'lbg': lg, # constrainst lower bound (basically giving == constraint)
            'ubg':  ug,
            'lbx': lbx, 
            'ubx': ubx
        }

        self.solver = solver
        self.args = args


        self.X0 = ca.DM.ones((n_alphas, 1))

        self.n_states = n_states
        self.n_alphas = n_alphas
        self.N = N

    def DM2Arr(self,dm):
    # returns a full matrix instead if a sparse one
        return np.array(dm.full())
    def get_estimate(self):
        
        df = pd.read_csv("power_data_intp1.csv", names=["pump_power", "input_omega"])
        arr = df.to_numpy(dtype=np.float32)
        #print(df.head())
        T_estim_start = self.T_estim_start
        N = self.N
        n_states = self.n_states
        Power_true = np.array(arr[T_estim_start:T_estim_start + N,0])
        Power_true = np.reshape(Power_true, (N,1))
        omega_true = np.array(arr[T_estim_start:T_estim_start + N,1])/100
        #print(omega_true)
        omega_true = np.reshape(omega_true, (N,1))
             
        #Power_true = np.array([3,7,13,21,31,43,57,73,91,111,133,157,183,211,241,273,307,343,381,421])
        print(len(Power_true))
        #omega_true = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        print(len(omega_true))
        #Power_0 = Power_true[0]
    
        start = time()
        
        
        
        X0 = self.X0
        args = self.args
        solver = self.solver
        n_states = self.n_states
        n_alphas = self.n_alphas
        N = self.N
        args['p'] = ca.vertcat(
            Power_true,
            omega_true,
        )

        print(args['lbx'])
        print(args['ubx'])

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
        
        sol_x = sol['x']
        #alpha_fit = sol['x'][-self.n_alphas:]
        print(sol_x)
        #print(alpha_fit)

        end = time()
        print(end - start, "s")

        return sol_x #, alpha_fit, gamma_fit
        
myEstimator = Estimator()
myEstimator.estimator_setup()
sol_x = myEstimator.get_estimate()
df = pd.read_csv("power_data_intp1.csv", names=["Power", "input_omega"])
arr = df.to_numpy(dtype=np.float32)
T_estim_start = myEstimator.T_estim_start
N = myEstimator.N
n_states = myEstimator.n_states
Power_true = np.array(arr[T_estim_start:T_estim_start + N,0])
Power_true = np.reshape(Power_true, (N,1))
#print(Power_true.shape)
omega_true = np.array(arr[T_estim_start:T_estim_start + N,1])/100
#print(omega_true)
omega_true = np.reshape(omega_true, (N,1))

power_model = []
#Power_true = np.array([3,7,13,21,31,43,57,73,91,111,133,157,183,211,241,273,307,343,381,421])
#omega_true = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
print(sol_x)
for omega in omega_true:
    power_fun = sol_x[0]*1 + sol_x[1]*omega + sol_x[2]*omega**2 + sol_x[3]*omega**3
    #print(omega[0],power_fun)
    power_model.append(power_fun.full().item())
    #print(type(power_fun.full()))
plt.plot(omega_true, Power_true)

plt.plot(omega_true, power_model)

plt.legend(["True", "Model"])
plt.show()

# IF ALPHA MIN=0.1, ALPHA MAX = 10 the optimizer sucks
# IF Alpha min = 0.9 alpha max = 1.1 it is great 
# 