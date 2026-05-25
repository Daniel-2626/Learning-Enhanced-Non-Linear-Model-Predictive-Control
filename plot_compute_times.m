%% Get data
logged_data_nominal = load("Traj_Simulation_Data/nominal_heating_T_env_plus_5_mismatched.mat");
outputs_nominal = logged_data_nominal.data;

cmp_time_nominal = getElement(outputs_nominal, "elapsed").Values.Data;

%% Get data
logged_data_nn = load("Deriv_Simulation_Data/adaptive_heating_mismatch.mat");
outputs_nn = logged_data_nn.data;
cmp_time_nn = getElement(outputs_nn, "elapsed").Values.Data;

%%
stem(cmp_time_nn)
hold on
stem(cmp_time_nominal)
yline(1000, 'LineWidth',2)
legend("NN+MPC", "MPC")
xlabel("Iteration")
ylabel("Computation time (ms)")

%% Filtering for NN times
% only NN training
idxs = [201, 401, 601, 801, 1001, 1201, 1401]
nn_training_times = cmp_time_nn(cmp_time_nn > 600);
nn_iteration_times = cmp_time_nn(cmp_time_nn < 600)
mean_training = mean(nn_training_times)
max_training = max(nn_training_times)

mean_iteration = mean(nn_iteration_times)
max_iteration = max(nn_iteration_times)
stem(nn_iteration_times)

%% 
nom_mean_iteration = mean(cmp_time_nominal)
nom_max_iteration = max(cmp_time_nominal)