%% Load data 
power_mat = readmatrix("power_data_intp1.csv");
power = power_mat(:,1)/1000;
omega = power_mat(:, 2)/100;
power_fun = (-0.21574 + 10.0501*omega.^2 + 6.84987*omega.^3)/1000;

%%
%%

figure(1)
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
plot(omega,power_fun, 'LineWidth',6, 'Color', RGB(2,:))
hold on
plot(omega,power, 'LineWidth',6, 'LineStyle',':', 'Color', 'black')

legend("Fitted power", "True power", 'Location','northwest')
xlabel("Scaled angular velocity")
ylabel('Power (kW)', 'Interpreter', 'latex')
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

