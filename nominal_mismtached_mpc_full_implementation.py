import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
from time import time
# setting matrix weights
Q_h_1 = 10
Q_h_2 = 10

R = 0.1

step_horizon = 0.5 # time between steps in seconds
N = 50 # number of look ahead steps

# Actual
A1 = 1
a1 = 0.1
A2 = 1
a2 = 0.1
rho = 1000
k = 1000
grav = 9.82

# Model
nominal_ratio = 0.7
A1_model = 1 * nominal_ratio
a1_model = 0.1 #/ nominal_ratio
A2_model = 1 * nominal_ratio
a2_model = 0.1 #* nominal_ratio
k_model = 1000#*nominal_ratio
rho_model = 1000*nominal_ratio

grav_model = 9.82

sim_time = 200


h_1_max = 2
h_2_max = 2
# params
h_1_init = 0.05
h_2_init = 0.05
h_1_target = 1
h_2_target = 1


u_max = 0.8
u_min = 0
u_in_ss = a1*np.sqrt(2*grav*h_1_target)
# Performs the shift after a control has been applied
def shift_timestep(step_horizon, t0, state_init, u, f):
    st, contr = state_init, u[:,0]
    
    K1 = f(st, contr)
    K2 = f(st + step_horizon/2 * K1, contr)
    K3 = f(st + step_horizon/2 * K2, contr)
    K4 = f(st + step_horizon * K3, contr)
   
    # Changes sparse matrix into a full matrix
    next_state = ca.DM.full(st + (step_horizon / 6) * (K1 + 2*K2 + 2*K3 + K4))

    t0 = t0 + step_horizon
    u0 = ca.horzcat(
            u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )
    # Note on reshape
    # With current arguments, have reshape(DM a, int nrow, int ncol)
    # Reshape reshapes a matrix into a certain dimension, e.g. let v be 1x6 matrix
    # Then reshape(v, 3, 2) returns a 3x2 matrix
    # If one argument is -1 then infers the size from the other (e.g. np.reshape(v, -1, 3) infers that nrow must be 2)
    # Here transforms last column of u into a column vector (idk if really necessary but maybe some double parenthesis issues)

    return t0, next_state, u0 # Will be set as initial condition in next iteration

def DM2Arr(dm):
    # returns a full matrix instead if a soarse ibe
    return np.array(dm.full())

# state symbolic variables
h_1 = ca.SX.sym('h_1')
h_2 = ca.SX.sym('h_2')
states = ca.vertcat(
    h_1,
    h_2
)
n_states = states.numel() # Returns amount of elements

# control symbolic variables
u_in = ca.SX.sym('u_in')

controls = u_in
n_controls = controls.numel()

# matrix containing all states over all time steps
X = ca.SX.sym('X', n_states, N+1)

# matrix containing all control actions over all time steps
U = ca.SX.sym('U', n_controls, N)

# column vector for storing initial state and target state
P = ca.SX.sym('P', n_states + n_states)

# State weights matrix
Q = ca.diagcat(Q_h_1, Q_h_2)

# Controls weight matrix
R = R # ca.diagcat(R1, R2, R3, R4)

# discretization model (e.g. x2 = f(x1, v, t) = x1 + v*dt)
## Replace with cascaded tanks
## Controller model (mismatched)
x1_dot_model = k_model*u_in/(rho_model*A1_model) - a1_model/A1_model *ca.sqrt(2*grav_model*h_1+0.001)
x2_dot_model = a1_model/A1_model *ca.sqrt(2*grav_model*h_1+0.001) - a2_model/A2_model *ca.sqrt(2*grav_model*h_2+0.001)
ode_model = ca.vertcat(x1_dot_model, x2_dot_model)

f_model = ca.Function("f_model", [states,controls], [ode_model], ["x","u"], ["ode"])
## Actual model
x1_dot = k*u_in/(rho*A1) - a1/A1 *ca.sqrt(2*grav*h_1+0.001)
x2_dot = a1/A1 *ca.sqrt(2*grav*h_1+0.001) - a2/A2 *ca.sqrt(2*grav*h_2+0.001)
ode = ca.vertcat(x1_dot, x2_dot)

f_actual = ca.Function("f_actual", [states,controls], [ode], ["x","u"], ["ode"])

cost_fn = 0 # Cost function
g = X[:, 0] - P[:n_states] # constraints in the equation, that x0 = p0 

# Runge Kutta
for k in range(N):
    # model update, what our controller knows
    st = X[:, k]
    contr = U[:, k]
    cost_fn = (cost_fn +
               (st-P[n_states:]).T @ Q @ (st-P[n_states:]) +
               contr.T @ R @ contr
               )
    st_next = X[:, k+1]
    K1 = f_model(st, contr)
    K2 = f_model(st + step_horizon/2 * K1, contr)
    K3 = f_model(st + step_horizon/2 * K2, contr)
    K4 = f_model(st + step_horizon * K3, contr)
    st_next_RK4 = st + (step_horizon / 6) * (K1 + 2*K2 + 2*K3 + K4)
    g = ca.vertcat(g, st_next - st_next_RK4) # Basically saying X[:, k+1] = runge kutta evaluation

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

lbx[0: n_states*(N+1): n_states] = 0 # Different notation to matlab. Here have [start:stop.step]. We fill out the vector related to states [start:n_states*(N+1)] with a lower bound every n_states, meaning we set lower bound for first state at each time step
lbx[1: n_states*(N+1): n_states] = 0 # Here set lower bound for second state

ubx[0: n_states*(N+1): n_states] = h_1_max 
ubx[1: n_states*(N+1): n_states] = h_2_max 

lbx[n_states*(N+1):] = u_min # same for all inputs
ubx[n_states*(N+1):] = u_max 

args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)), # constrainst lower bound (basically giving == constraint)
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),
    'lbx': lbx, 
    'ubx': ubx
}

t0 = 0
state_init = ca.DM([h_1_init, h_2_init]) # initial state
state_target = ca.DM([h_1_target, h_2_target])

t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))
X0 = ca.repmat(state_init, 1, N+1)

mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:,0])
times = np.array([[0]])


if __name__ == '__main__':
    main_loop = time()
    while (mpc_iter * step_horizon < sim_time):
        t1 = time()
        args['p'] = ca.vertcat(
            state_init, # current state
            state_target 
        )

        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),
            ca.reshape(u0, n_controls*N, 1)
        )

        sol = solver(
            x0 = args['x0'],
            lbx = args['lbx'],
            ubx = args['ubx'],
            lbg = args['lbg'],
            ubg = args['ubg'],
            p = args['p']
        )

        u = ca.reshape(sol['x'][n_states*(N+1):], n_controls, N   ) # gives u as a vector where rows are the different inputs and columns the time steps (applied input at col 0)
        X0 = ca.reshape(sol['x'][:n_states*(N+1)], n_states, N+1)
        #print(X0)
        # if fewer than 3 states this can probably be changed to vstack or hstack
        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))
        # in case more than one control, this probably makes sense to change to dstack
        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))

        t = np.vstack((
            t,
            t0
        ))
        # In code, this is the actual model update
        t0, state_init, u0 = shift_timestep(step_horizon, t0, state_init, u, f_actual)

        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )

        t2 = time()
        print("iter", mpc_iter)
        print("time", t2-t1)
        times = np.vstack((
            times, 
            t2-t1
        ))

        mpc_iter += 1

    main_loop_time = time()
    ss_error = ca.norm_2(state_init - state_target)

    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')
    print('final error: ', ss_error)

h_1_num = cat_states[0, 0, :]
h_2_num = cat_states[1, 0, :]
u_in_num = cat_controls
print("input for steady state at reference:", a1*np.sqrt(2*grav*h_1_target))
#u_in_num = cat_controls
fig, (ax1, ax2) = plt.subplots(2,1,sharey=False)
ax1.plot(h_1_num)
ax1.plot(h_2_num)
ax1.axhline(h_1_target, xmax=2000, color = 'black')
ax1.set_title("States")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Height (m)")
ax1.legend(["$h_1$", "$h_2$", "$h_{ref}$"])
ax2.plot(u_in_num)
ax2.axhline(u_in_ss, color= 'black')
ax2.legend(["$u_{in}$", "$u_{in,ss}$"])
ax2.set_title("Input")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Voltage (V)")
fig.suptitle("Nominal controller operating with mismatched (70%) model information", fontsize=16)

plt.show()