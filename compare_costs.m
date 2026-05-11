%% Simulator set-up
N_reps = 3;

%% Get data
logged_data_traditional = load("Simulation_Data/traditional_new_scaling_mismatched_new_loss_filtering_alpha_train_often_tanh_4_input.mat");
outputs_traditional = logged_data_traditional.data;

time_traditional = getElement(outputs_traditional, "time").Values.Data;
omega_traditional = getElement(outputs_traditional, "input_omega").Values.Data;
Q_heat_traditional = getElement(outputs_traditional, "input_q_heat").Values.Data;

pump_power_traditional = getElement(outputs_traditional, "pump_power").Values.Data;
heatingPwr_traditional = abs(getElement(outputs_traditional, "heatingPwr").Values.Data);
T_bat_traditional = getElement(outputs_traditional, "Pack3").Values.Data;

%% Get data
logged_data_nn = load("Simulation_Data/adaptive_new_scaling_mismatched_new_loss_filtering_alpha_train_often_tanh_3_input.mat");
outputs_nn = logged_data_nn.data;
omega_nn = getElement(outputs_nn, "input_omega").Values.Data;
Q_heat_nn = getElement(outputs_nn, "input_q_heat").Values.Data;

time_nn = getElement(outputs_nn, "time").Values.Data;
pump_power_nn = getElement(outputs_nn, "pump_power").Values.Data;
heatingPwr_nn = abs(getElement(outputs_nn, "heatingPwr").Values.Data);
T_bat_nn = getElement(outputs_nn, "Pack3").Values.Data;

%% Interpolation
dt = 5;
t = 1:dt:N_reps*2474;
T_bat_traditional_interp = interp1(time_traditional, T_bat_traditional, t);

T_bat_nn_interp = interp1(time_nn, T_bat_nn, t);

%%
plot(T_bat_traditional_interp,'LineWidth',2)
hold on
plot(T_bat_nn_interp, 'LineWidth',2)
yline(20.5, 'LineWidth',2)
legend('T_{bat,nom}', 'T_{bat,nn}', 'Target', 'Location','southeast')
xlabel("Time (s)")
ylabel(['Temperature (C' char(176) ')'])
axis([1, N_reps*2475/5, 17, 23])


%% Cost function
Q = 10*100^2;
R_omega = 0.1;
R_Q_heat = 1;
cmp_start = 200; % When NN is turned on
omega_traditional = omega_traditional(cmp_start:end)/100;
Q_heat_traditional = Q_heat_traditional(cmp_start:end)/1000;
T_bat_traditional = ((T_bat_traditional_interp(cmp_start:end)-20.5)/100)';


omega_nn = omega_nn(cmp_start:end)/100;
Q_heat_nn = Q_heat_nn(cmp_start:end)/1000;
T_bat_nn = ((T_bat_nn_interp(cmp_start:end)-20.5)/100)';

cost_traditional = omega_traditional' * R_omega * omega_traditional + Q_heat_traditional'*R_Q_heat*Q_heat_traditional + T_bat_traditional'*Q*T_bat_traditional;
cost_nn = omega_nn' * R_omega * omega_nn + Q_heat_nn'*R_Q_heat*Q_heat_nn + T_bat_nn'*Q*T_bat_nn;