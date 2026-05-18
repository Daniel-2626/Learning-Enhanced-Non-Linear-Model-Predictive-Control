%% Get data
logged_data = load("Simulation_Data/adaptive_mpc_for_benchmark_t_env_17_steady_state_check.mat");
outputs = logged_data.data;

omega_ss = getElement(outputs, "omega_ss").Values.Data;
Q_heat_ss = getElement(outputs, "Q_ss").Values.Data;


%% Get data
logged_data_residual_update = load("Simulation_Data/adaptive_mpc_for_benchmark_t_env_17_steady_state_check_residual_update.mat");
outputs_residual_update = logged_data_residual_update.data;
omega_ss_residual_update = getElement(outputs_residual_update, "omega_ss").Values.Data;
Q_heat_ss_residual_update = getElement(outputs_residual_update, "Q_ss").Values.Data;


%%
figure(1)
plot(omega_ss_residual_update, 'LineWidth',2)
hold on
plot(omega_ss, 'LineWidth',2, 'LineStyle','--')
%yline(1000, 'LineWidth',2)
legend("NN+MPC residual update", "NN+MPC")
xlabel("Iteration")
ylabel("Computation time (ms)")

%%
figure(2)
plot(Q_heat_ss_residual_update, 'LineWidth',2)
hold on
plot(Q_heat_ss, 'LineWidth',2, 'LineStyle','--')
%yline(1000, 'LineWidth',2)
legend("NN+MPC residual update", "NN+MPC")
xlabel("Iteration")
ylabel("Computation time (ms)")