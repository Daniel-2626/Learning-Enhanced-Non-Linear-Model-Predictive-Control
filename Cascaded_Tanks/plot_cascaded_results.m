
adaptive_results = readtable("cascaded_adaptive_mismatched.csv");
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
nn_on = 20*dt+dt
%% Plotting

%% h1
figure(1)
clf
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');


plot(times, h1_ada, 'LineWidth',2.5)
hold on
plot(times, h1_nom, 'LineWidth',2.5)
ylabel("Water level tank 1 (m)");
xlabel("Time (s)");
yline(h_ref, '--', 'LineWidth', 2.5)
xl = xline(nn_on, '-',{'Neural network on'}, 'LineWidth', 2.5);
%xl.LabelHorizontalAlignment = 'center';
xl.LabelVerticalAlignment = 'bottom';
xlim([0 rows*dt-dt])
ylim([0.7 1.15])
legend('Adaptive', 'Nominal', 'Reference')
%ylabel('Height Tank 1 (m)');
%xlabel('Time (s)');
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

%% u
figure(2)
plot(times, u_ada, 'LineWidth',2)
hold on
plot(times, u_nom, 'LineWidth',2)