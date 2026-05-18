
adaptive_results = readtable("cascaded_adaptive_mismatched_deriv.csv");
nominal_results = readtable("cascaded_nominal_mismatched.csv");

u_nom =nominal_results.u;
h1_nom = nominal_results.h1;
h2_nom = nominal_results.h2;

u_ada =adaptive_results.u;
h1_ada = adaptive_results.h1;
h2_ada = adaptive_results.h2;

[rows, ~] = size(u_nom);
dt = 0.5;
times = 0:dt:rows*dt-dt;

h_ref = 1;
nn_on = 25 + dt
%% Plotting

%% h1
figure(1)
clf
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');


plot(times, h1_ada, 'LineWidth',6)
hold on
plot(times, h1_nom, '-- ', 'LineWidth',6)
ylabel("Water level tank 1 (m)");
xlabel("Time (s)");
yline(h_ref, '--', 'LineWidth', 6)
xl = xline(nn_on, '-',{'Neural network on'}, 'LineWidth',6);
%xl.LabelHorizontalAlignment = 'center';
xl.LabelVerticalAlignment = 'bottom';
xlim([20 80])
ylim([0.7 1.15])
legend('Derivative loss', 'Nominal', 'Reference')
%ylabel('Height Tank 1 (m)');
%xlabel('Time (s)');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

%% h2
figure(2)
clf
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');


plot(times, h2_ada, 'LineWidth',6)
hold on
plot(times, h2_nom, '--', 'LineWidth',6)
ylabel("Water level tank 2 (m)");
xlabel("Time (s)");
yline(h_ref, '--', 'LineWidth', 6)
xl = xline(nn_on, '-',{'Neural network on'}, 'LineWidth', 6);
%xl.LabelHorizontalAlignment = 'center';
xl.LabelVerticalAlignment = 'bottom';
xlim([20 80])
ylim([0.7 1.15])
legend('Derivative loss', 'Nominal', 'Reference')
%ylabel('Height Tank 1 (m)');
%xlabel('Time (s)');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');


%% u
figure(3)
clf
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');


plot(times, u_ada, 'LineWidth',6)
hold on
plot(times, u_nom, '--', 'LineWidth',6)
ylabel("Input (V)");
xlabel("Time (s)");
%yline(h_ref, '--', 'LineWidth', 2.5)
xl = xline(nn_on, '-',{'Neural network on'}, 'LineWidth', 6);
%xl.LabelHorizontalAlignment = 'center';
xl.LabelVerticalAlignment = 'bottom';
xlim([20 80])
ylim([0 0.9])
legend('Derivative loss', 'Nominal')
%ylabel('Height Tank 1 (m)');
%xlabel('Time (s)');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
