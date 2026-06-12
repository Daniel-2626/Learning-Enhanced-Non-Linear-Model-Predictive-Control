%% Simulator set-up
N_reps = 3;
CELSIUS_TO_KELVIN = 273.15
%% Get data
logged_data_traditional = load("Deriv_Simulation_Data/nominal_cooling_mismatch.mat");
outputs_traditional = logged_data_traditional.data;

time_traditional = getElement(outputs_traditional, "time").Values.Data;
omega_traditional = getElement(outputs_traditional, "input_omega").Values.Data;
Q_heat_traditional = getElement(outputs_traditional, "input_q_heat").Values.Data;

pump_power_traditional = getElement(outputs_traditional, "pump_power").Values.Data;
heatingPwr_traditional = abs(getElement(outputs_traditional, "heatingPwr").Values.Data);
T_bat_traditional = getElement(outputs_traditional, "Pack3").Values.Data + CELSIUS_TO_KELVIN;

%% Get data
logged_data_nn = load("Deriv_Simulation_Data/adaptive_cooling_mismatch_fixed_nonlin.mat");
outputs_nn = logged_data_nn.data;
omega_nn = getElement(outputs_nn, "input_omega").Values.Data;
Q_heat_nn = getElement(outputs_nn, "input_q_heat").Values.Data;

time_nn = getElement(outputs_nn, "time").Values.Data;
pump_power_nn = getElement(outputs_nn, "pump_power").Values.Data;
heatingPwr_nn = abs(getElement(outputs_nn, "heatingPwr").Values.Data);
T_bat_nn = getElement(outputs_nn, "Pack3").Values.Data + CELSIUS_TO_KELVIN;

%% Interpolation
dt = 5;
t = 1:dt:N_reps*2474;
T_bat_traditional_interp = interp1(time_traditional, T_bat_traditional, t);

T_bat_nn_interp = interp1(time_nn, T_bat_nn, t);
nn_on = 1000;
%%
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
plot(t,T_bat_nn_interp-273.15, 'LineWidth',6)
hold on
plot(t,T_bat_traditional_interp-273.15,'LineWidth',6)
yline(20.5, '--', 'LineWidth',6)
xl = xline(nn_on, '-',{'Neural network on'}, 'LineWidth',6, 'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'left');

legend('Adaptive', 'Nominal', 'Target', 'Location','northeast')
xlabel("Time (s)")
ylabel('Temperature ($^\circ$C)', 'Interpreter', 'latex')
axis([1, N_reps*2475, 19.5, 26])
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

%%
% set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
% set(groot, 'defaultTextInterpreter', 'latex');
% set(groot, 'defaultLegendInterpreter', 'latex');
% plot(t,Q_heat_nn, 'LineWidth',3)
% hold on
% plot(t,Q_heat_traditional,'LineWidth',3)
% yline(0, '--', 'LineWidth',3)
% xl = xline(nn_on, '-',{'Neural network on'}, 'LineWidth',6, 'LabelVerticalAlignment', 'middle', 'LabelHorizontalAlignment', 'left');
% 
% legend('Power, adaptive', 'Power, nominal', 'Location','northwest')
% xlabel("Time (s)")
% ylabel('Power (W)', 'Interpreter', 'latex')
% axis([3000, 7000, -4000, 2000])
% set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
% set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
%% Cost function
Q = 10*100^2;
R_omega = 1;
R_Q_heat = 10;
cmp_start = 200; % When NN is turned on
tracking = 273.15 + 20.5;
omega_traditional = omega_traditional(cmp_start:end)/100;
Q_heat_traditional = Q_heat_traditional(cmp_start:end)/1000;
T_bat_traditional = ((T_bat_traditional_interp(cmp_start:end)-tracking)/100)';


omega_nn = omega_nn(cmp_start:end)/100;
Q_heat_nn = Q_heat_nn(cmp_start:end)/1000;
T_bat_nn = ((T_bat_nn_interp(cmp_start:end)-tracking)/100)';

cost_traditional = omega_traditional' * R_omega * omega_traditional + Q_heat_traditional'*R_Q_heat*Q_heat_traditional + T_bat_traditional'*Q*T_bat_traditional;
cost_nn = omega_nn' * R_omega * omega_nn + Q_heat_nn'*R_Q_heat*Q_heat_nn + T_bat_nn'*Q*T_bat_nn;

%% Reference error
cmp_start = 502;
T_bat_traditional = ((T_bat_traditional_interp(cmp_start:end)-tracking));
[~, nvals] = size(T_bat_traditional)
RMSE_Traditional = sqrt(sum(T_bat_traditional.^2)/nvals)
T_bat_nn = ((T_bat_nn_interp(cmp_start:end)-tracking));
RMSE_nn = sqrt(sum(T_bat_nn.^2)/nvals)

%cost_traditional
%cost_nn

