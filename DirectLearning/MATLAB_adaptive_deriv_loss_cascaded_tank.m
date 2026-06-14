
%% importing CasADi
addpath(genpath('<PATH_TO_CASADI_INSTALL>'));

%% Importing python function
train_network = py.importlib.import_module('train_network');
py.importlib.reload(train_network)
%% Parameters for model and MPC
% setting matrix weights
Q_h_1 = 10;
Q_h_2 = 10;

R = 0.1;

step_horizon = 0.5; % time between steps in seconds
N = 50; % number of look ahead steps

% Actual
A1 = 1;
a1 = 0.1;
A2 = 1;
a2 = 0.1;
rho = 1000;
k = 1000;
grav = 9.82;

% Model
nominal_ratio = 0.7;
A1_model = 1; % * nominal_ratio;
a1_model = 0.1 * nominal_ratio;
A2_model = 1; %* nominal_ratio;
a2_model = 0.1 * nominal_ratio;
k_model = 1000;%*nominal_ratio
rho_model = 1000;%*nominal_ratio;

grav_model = 9.82;

sim_time = 80;


h_1_max = 2;
h_2_max = 2;
% params
h_1_init = 0.5;
h_2_init = 0.5;
h_1_target = 1;
h_2_target = 1;


u_max = 0.8;
u_min = 0;
u_in_ss = a1*sqrt(2*grav*h_1_target);

% state symbolic variables
h_1 = casadi.SX.sym('h_1');
h_2 = casadi.SX.sym('h_2');
states = [h_1;h_2];
n_states = 2; %states.numel() % Returns amount of elements

% control symbolic variables
u_in = casadi.SX.sym('u_in');


%% MLP symbolic set-up
input_dim = 3;
output_dim = 2;
hidden_dim = 16;

nn_on = casadi.SX.sym('nn_on', 1);

% Input layer
x = [h_1; h_2; u_in];
% An MLP with 2 hidden layers
% First hidden layer 
W_1 = casadi.SX.sym('weights_1', hidden_dim, input_dim);
b_1 = casadi.SX.sym('bias_1', hidden_dim, 1);
% Second hidden layer
W_2 = casadi.SX.sym('weights_2', hidden_dim, hidden_dim);
b_2 = casadi.SX.sym('bias_2', hidden_dim, 1);
% Output layer
W_3 = casadi.SX.sym('weights_3', output_dim, hidden_dim);
b_3 = casadi.SX.sym('bias_3', output_dim, 1);

y_1 = tanh( W_1*x + b_1);
y_2 = tanh( W_2*y_1 + b_2);
y_3 =  W_3*y_2 + b_3;


MLP = y_3;

parameters = [reshape(W_1, 1,hidden_dim*input_dim), b_1',...
    reshape(W_2, 1,hidden_dim*hidden_dim), b_2', ...
    reshape(W_3, 1,hidden_dim*output_dim), b_3'];


%%
controls = u_in;
n_controls = 1; %controls.numel()
[~, n_parameters] = size(parameters);
% matrix containing all states over all time steps
X = casadi.SX.sym('X', n_states, N+1);

% matrix containing all control actions over all time steps
U = casadi.SX.sym('U', n_controls, N);

% column vector for storing initial state and target state, nn parameters
% and nn on switch (+1) 
P = casadi.SX.sym('P', n_states + n_states + n_parameters + 1);

% State weights matrix
Q = diag([Q_h_1, Q_h_2]);

% Controls weight matrix
R = R; % casadi.diagcasadit(R1, R2, R3, R4)



%% Controller model (mismatched)
x1_dot_nominal = k_model*u_in/(rho_model*A1_model) - a1_model/A1_model *sqrt(2*grav_model*h_1+0.001);
x2_dot_nominal = a1_model/A1_model *sqrt(2*grav_model*h_1+0.001) - a2_model/A2_model *sqrt(2*grav_model*h_2+0.001);
ode_nominal = [x1_dot_nominal; x2_dot_nominal];

f_nominal = casadi.Function('f_nominal', {states,controls}, {ode_nominal}, {'x','u'}, {'ode_nominal'});
f_residual = casadi.Function('f_model', {states,controls, parameters}, {MLP}, {'x','u', 'p'}, {'ode_res'});

ode_model = f_nominal(states, controls) + nn_on*f_residual(states, controls, parameters);

f_model = casadi.Function('f_model', {states,controls, parameters, nn_on}, {ode_model}, {'x','u', 'p', 'nn_on'}, {'ode_model'});

%% Actual model
x1_dot = k*u_in/(rho*A1) - a1/A1 *sqrt(2*grav*h_1+0.001);
x2_dot = a1/A1 *sqrt(2*grav*h_1+0.001) - a2/A2 *sqrt(2*grav*h_2+0.001);
ode = [x1_dot; x2_dot];

f_actual = casadi.Function('f_actual', {states,controls}, {ode}, {'x','u'}, {'ode'});

cost_fn = 0; % Cost function
g = X(:, 1) - P(1:n_states); % constraints in the equation, that x0 = p0 

% Runge Kutta
for k = 1:N
    % model update, what our controller knows
    st = X(:, k);
    contr = U(:, k);
    cost_fn = (cost_fn + ...
               transpose(st-P(n_states+1:n_states + n_states)) * Q * (st-P(n_states+1:n_states + n_states)) + ...
               transpose(contr) * R * contr);
    st_next = X(:, k+1);
    K1 = f_model(st, contr, P(n_states + n_states + 1:end-1), P(end));
    K2 = f_model(st + step_horizon/2 * K1, contr, P(n_states + n_states + 1:end-1), P(end));
    K3 = f_model(st + step_horizon/2 * K2, contr, P(n_states + n_states + 1:end-1), P(end));
    K4 = f_model(st + step_horizon * K3, contr, P(n_states + n_states + 1:end-1), P(end));
    st_next_RK4 = st + (step_horizon / 6) * (K1 + 2*K2 + 2*K3 + K4);
    g = [g; st_next - st_next_RK4]; % Basicasadilly saying X[:, k+1] = runge kutta evaluation
end
OPT_variables = [
    reshape(X, (N+1)*(n_states),1);  
    reshape(U, N,1)   
];
nlp_prob = struct('f', cost_fn,'x', OPT_variables,'g', g,'p', P);

opts = struct(...
    'ipopt',  struct(...
        'max_iter', 2000,...
        'print_level', 0,...
        'acceptable_tol', 1e-8,...
        'acceptable_obj_change_tol', 1e-6,...
        'hessian_approximation', 'limited-memory'...
    ),...
    'print_time', 0 ...
);

solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, opts);

lbx = casadi.DM.zeros(n_states*(N+1) + n_controls*N, 1);
ubx = casadi.DM.zeros(n_states*(N+1) + n_controls*N, 1); % long column vector 

lbx(1: n_states: n_states*(N+1)+1) = 0; % Different notation to matlab. Here have [start:stop.step]. We fill out the vector related to states [start:n_states*(N+1)] with a lower bound every n_states, meaning we set lower bound for first state at each time step
lbx(2: n_states: n_states*(N+1)+1) = 0; % Here set lower bound for second state

ubx(1: n_states: n_states*(N+1)+1) = h_1_max; 
ubx(2:  n_states: n_states*(N+1)+1) = h_2_max;

lbx(n_states*(N+1)+1:end) = u_min; % same for all inputs
ubx(n_states*(N+1)+1:end) = u_max;

args = struct(...
    'lbg', casadi.DM.zeros(n_states*(N+1), 1), ...% constrainst lower bound (basicasadilly giving == constraint)
    'ubg', casadi.DM.zeros(n_states*(N+1), 1),...
    'lbx', lbx, ...
    'ubx', ubx...
);

t0 = 0;
state_init = [h_1_init, h_2_init]; % initial state
state_target = [h_1_target, h_2_target];

t = casadi.DM(t0);

u0 = casadi.DM.zeros(n_controls, N);
X0 = repmat(state_init, 1, N+1);

mpc_iter = 1;
traj = zeros(n_states, sim_time+1);
traj(:, 1) = state_init';
inputs = zeros(n_controls, sim_time);
%state_dict = weights; %1/40 * (2*rand(1, n_parameters)-1); % zeros(1, n_parameters);
nn_on = 0;
weights = zeros(1,n_parameters);
batchSize = 20;
residual_data = zeros(batchSize, n_states);
input_data = zeros(batchSize, n_states + n_controls);
T_update = batchSize;
evaluating_times = zeros(sim_time/step_horizon-1, 1);
while (mpc_iter * step_horizon < sim_time)
    tic
    args.p = [state_init,state_target, weights, nn_on
    ];
    % optimization variable current state
    args.x0 = [
        reshape(X0, n_states*(N+1), 1);...
        reshape(u0, n_controls*N, 1)
    ];
    sol = solver(...
        'x0', args.x0,...
        'lbx', args.lbx,...
        'ubx', args.ubx,...
        'lbg', args.lbg,...
        'ubg',args.ubg,...
        'p', args.p);


    u = reshape(sol.x(n_states*(N+1)+1:end), n_controls, N); % gives u as a vector where rows are the different inputs and columns the time steps (applied input at col 0)
    X0 = reshape(sol.x(1:n_states*(N+1)), n_states, N+1);
    % if fewer than 3 states this can probably be changed to vstack or hstack
    
    % In code, this is the actual model update
    [t0, state_init, u0] = shift_timestep(step_horizon, t0, state_init, u, f_actual);
    
    residual = f_actual(state_init, u(:,1)) - f_nominal(state_init, u(:,1));
    data_idx = mod(mpc_iter, batchSize); 
    if data_idx==0
        data_idx = batchSize;
    end
    residual_data(data_idx, :) = full(residual');
    input_data(data_idx, :) =  [full(state_init), full(u(:,1))];
    if mod(mpc_iter,T_update) == 0
        nn_on = 1;
        weights = double(train_network.train_network(residual_data, input_data));
        %stop
    end
    X0 = [...
        X0(:, 2:end),...
        reshape(X0(:, end), 2, 1)...
    ];
    evaluating_times(mpc_iter) = toc;

    inputs(:, mpc_iter) = full(u(:,1));
    disp("iteration:")
    disp(mpc_iter)
    mpc_iter = mpc_iter + 1;
    traj(:, mpc_iter) = full(state_init');

end
ss_error = norm(state_init - state_target)

plot(traj(1,:), LineWidth=2)
hold on
plot(traj(2,:), LineWidth=2)
yline(h_1_target, LineWidth=2, LineStyle="--")
legend("h_1", "h_2", "reference")
%%

% Performs the shift after a control has been applied
function [t0, next_state, u0] = shift_timestep(step_horizon, t0, state_init, u, f)
    st = state_init';
    contr = u(:,1);
    K1 = f(st, contr);
    K2 = f(st + step_horizon/2 * K1, contr);
    K3 = f(st + step_horizon/2 * K2, contr);
    K4 = f(st + step_horizon * K3, contr);
   
    % Changes sparse matrix into a full matrix
    next_state = (st + (step_horizon / 6) * (K1 + 2*K2 + 2*K3 + K4))';

    t0 = t0 + step_horizon;
    u0 = [u(:, 2:end),...
        reshape(u(:, end), 1, 1)...
    ];
    % Note on reshape
    % With current arguments, have reshape(DM a, int nrow, int ncol)
    % Reshape reshapes a matrix into a certain dimension, e.g. let v be 1x6 matrix
    % Then reshape(v, 3, 2) returns a 3x2 matrix
    % If one argument is -1 then infers the size from the other (e.g. np.reshape(v, -1, 3) infers that nrow must be 2)
    % Here transforms last column of u into a column vector (idk if really necessary but maybe some double parenthesis issues)

end
function mat = DM2Arr(dm)
    % returns a full matrix instead if a soarse ibe
    mat = dm.full();
end 