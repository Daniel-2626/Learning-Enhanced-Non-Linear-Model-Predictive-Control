
simulation_results = readtable("simulation_results_mismatched.csv");

T_true =simulation_results.true - 273.15;
T_nominal = simulation_results.nominal-273.15;
T_adaptive = simulation_results.adaptive-273.15;


[rows, ~] = size(T_true);
dt = 5;
times = dt*200:dt:dt*200+ rows*dt-dt;

%% Plotting

%% T_bat pred
figure(1)
clf
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');


plot(times, T_true, 'LineWidth',6)
hold on
plot(times, T_nominal, 'LineWidth',6)
plot(times, T_adaptive,'LineWidth',6)

xlabel("Time (s)")
ylabel('Temperature ($^\circ$C)', 'Interpreter', 'latex')

%yline(h_ref, '--', 'LineWidth', 6)
%xl = xline(nn_on, '-',{'Neural network on'}, 'LineWidth',6);
%xl.LabelHorizontalAlignment = 'center';
%xl.LabelVerticalAlignment = 'bottom';
%xlim([20 80])
%ylim([0.7 1.15])
legend('True temperature', 'Nominal simulation', 'Adaptive simulation', 'Location','northwest')
%ylabel('Height Tank 1 (m)');
%xlabel('Time (s)');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

%% Load jacobian data
jacobians = readtable("jacobian.csv");
q_range = jacobians.q_range;
res_val_q = jacobians.res_val_q;
d_res_dq = jacobians.d_res_dq;
w_range = jacobians.w_range;
res_val_w = jacobians.res_val_w;
d_res_dw = jacobians.d_res_dw;
i_range = jacobians.i_range;
res_val_i = jacobians.res_val_i;
d_res_di = jacobians.d_res_di;
T_range = jacobians.T_range;
res_val_T = jacobians.res_val_T;
d_res_dT = jacobians.d_res_dT;

%%
figure(2)
clf
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

% Left y-axis
yyaxis left
plot(q_range, res_val_q, 'b', 'LineWidth', 6);
xlabel('Scaled heating power');
ylabel('Residual Value', 'Color', 'b');
ax = gca;
ax.YColor = 'b';

% Right y-axis
yyaxis right
plot(q_range, d_res_dq, 'r', 'LineWidth', 6);
ylabel('Jacobian (Sensitivity)', 'Color', 'r');
ax.YColor = 'r';

% Optional grid
grid on
%ylabel('Height Tank 1 (m)');
%xlabel('Time (s)');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

%%
figure(3)
clf
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');


%figure('Position', [100, 100, 1000, 600]);

% Left y-axis
yyaxis left
plot(w_range, res_val_w, 'b', 'LineWidth', 6);
xlabel('Scaled angular velocity');
ylabel('Residual Value', 'Color', 'b');
ax = gca;
ax.YColor = 'b';

% Right y-axis
yyaxis right
plot(w_range, d_res_dw, 'r', 'LineWidth', 6);
ylabel('Jacobian (Sensitivity)', 'Color', 'r');
ax.YColor = 'r';

% Optional grid
grid on
%ylabel('Height Tank 1 (m)');
%xlabel('Time (s)');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

%%
figure(4)
clf
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');


%figure('Position', [100, 100, 1000, 600]);

% Left y-axis
yyaxis left
plot(i_range, res_val_i, 'b', 'LineWidth', 6);
xlabel('Scaled current');
ylabel('Residual Value', 'Color', 'b');
ax = gca;
ax.YColor = 'b';

% Right y-axis
yyaxis right
plot(i_range, d_res_di, 'r', 'LineWidth', 6);
ylabel('Jacobian (Sensitivity)' , 'Color', 'r');
ax.YColor = 'r';

% Optional grid
grid on
%ylabel('Height Tank 1 (m)');
%xlabel('Time (s)');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

%%
figure(5)
clf
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');


%figure('Position', [100, 100, 1000, 600]);

% Left y-axis
yyaxis left
plot(T_range, res_val_T, 'b', 'LineWidth', 6);
xlabel('Scaled temperature');
ylabel('Residual Value', 'Color', 'b');
ax = gca;
ax.YColor = 'b';

% Right y-axis
yyaxis right
plot(T_range, d_res_dT, 'r', 'LineWidth', 6);
ylabel('Jacobian (Sensitivity)' , 'Color', 'r');
ax.YColor = 'r';

% Optional grid
grid on
%ylabel('Height Tank 1 (m)');
%xlabel('Time (s)');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');