%% Get data
logged_data_nominal = load("Simulation_Data/nominal_mpc_for_benchmark_t_env_-10_mismatched_model_N_200.mat");
outputs_nominal = logged_data_nominal.data;

cmp_time_nominal = getElement(outputs_nominal, "elapsed").Values.Data;

%% Get data
logged_data_nn = load("Simulation_Data/adaptive_mpc_for_benchmark_t_env_-10_mismatched_model_N_200.mat");
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
