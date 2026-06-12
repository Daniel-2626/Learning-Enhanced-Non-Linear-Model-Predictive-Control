%% Simulator set-up
N_reps = 3;
CELSIUS_TO_KELVIN = 273.15;
%% Get data
logged_data_traditional = load("nominal_cooling_plus_36_matched.mat");
outputs_traditional = logged_data_traditional.data;

time_traditional = getElement(outputs_traditional, "time").Values.Data;
omega_traditional = getElement(outputs_traditional, "input_omega").Values.Data;
Q_heat_traditional = getElement(outputs_traditional, "input_q_heat").Values.Data;
pump_power_traditional = getElement(outputs_traditional, "pump_power").Values.Data;
heatingPwr_traditional = abs(getElement(outputs_traditional, "heatingPwr").Values.Data);
T_bat_traditional = getElement(outputs_traditional, "Pack3").Values.Data + CELSIUS_TO_KELVIN;


%% Interpolation
dt = 5;
t = 1:dt:N_reps*2474;
T_bat_traditional_interp = interp1(time_traditional, T_bat_traditional, t);


%%
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

plot(t,T_bat_traditional_interp-273.15,'LineWidth',6, 'Color', '#DD5400')
yline(20.5, '--', 'LineWidth',6)

legend('Nominal', 'Target', 'Location','northeast')
xlabel("Time (s)")
ylabel('Temperature ($^\circ$C)', 'Interpreter', 'latex')
axis([1, N_reps*2475, 19, 36])
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
