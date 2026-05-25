%% Simulator set-up
N_reps = 3;
CELSIUS_TO_KELVIN = 273.15
%% Get data
data = readmatrix("nominal_sim.csv");
T_true = data(:,1);
T_true = [12.5+CELSIUS_TO_KELVIN; T_true];
T_sim = data(:,2);
T_sim = [12.5+CELSIUS_TO_KELVIN; T_sim];
t = 1:5:1000;
%%
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
plot(t,T_true-CELSIUS_TO_KELVIN, 'LineStyle', '-', 'LineWidth',6)
hold on
plot(t, T_sim-CELSIUS_TO_KELVIN, 'LineStyle', '--', 'LineWidth',6)

legend('True temperature', 'Nominal simulation', 'Location','southeast')
xlabel("Time (s)")
ylabel('Temperature ($^\circ$C)', 'Interpreter', 'latex')
axis([1, 1000, 10, 22])
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

max(abs(T_true - T_sim))
pred_errors = abs(T_true-T_sim);
RMSE = sqrt(sum(pred_errors.^2)/200);
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

