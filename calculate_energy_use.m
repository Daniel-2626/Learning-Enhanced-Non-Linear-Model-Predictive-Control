%% Simulator set-up
N_reps = 5;

%% Get data
logged_data_traditional = load("Simulation_Data/thermal_management_energy_cooling_matched_sg_3_input.mat");
outputs_traditional = logged_data_traditional.data;

time_traditional = getElement(outputs_traditional, "time").Values.Data;
omega_traditional = getElement(outputs_traditional, "input_omega").Values.Data;
pump_power_traditional = getElement(outputs_traditional, "pump_power").Values.Data;
heatingPwr_traditional = abs(getElement(outputs_traditional, "heatingPwr").Values.Data);
T_bat_traditional = getElement(outputs_traditional, "Pack3").Values.Data;

%% Get data
logged_data_nn = load("Simulation_Data/adaptive_thermal_management_energy_cooling_matched_sg_3_input.mat");
outputs_nn = logged_data_nn.data;
omega_nn = getElement(outputs_nn, "input_omega").Values.Data;
time_nn = getElement(outputs_nn, "time").Values.Data;
pump_power_nn = getElement(outputs_nn, "pump_power").Values.Data;
heatingPwr_nn = abs(getElement(outputs_nn, "heatingPwr").Values.Data);
T_bat_nn = getElement(outputs_nn, "Pack3").Values.Data;


%% Interpolation
dt = 1;
t = 1:dt:N_reps*2474;
pump_power_traditional_interp = interp1(time_traditional, pump_power_traditional, t);

T_bat_traditional_interp = interp1(time_traditional, T_bat_traditional, t);

pump_power_nn_interp = interp1(time_nn, pump_power_nn, t);
time_mpc_heat_pwr = 1:5:N_reps*2474;

T_bat_mpc_nn_interp = interp1(time_nn, T_bat_nn, t);

%% Riemman sum
hour_in_sec = 1/3600;

heating_energy_traditional = 5*hour_in_sec*sum(heatingPwr_traditional)/1000; % Heating power in Watts, convert to kW
pump_energy_traditional = hour_in_sec*sum(pump_power_traditional_interp);
energy_traditional = heating_energy_traditional + pump_energy_traditional;

heating_energy_nn = 5*hour_in_sec*sum(heatingPwr_nn)/1000; % heating power set every 5 s
pump_energy_nn = hour_in_sec*sum(pump_power_nn_interp);
energy_nn = heating_energy_nn + pump_energy_nn;

%%
figure(1)
plot(t,T_bat_mpc_nn_interp, 'LineWidth',6)
hold on
plot(t,T_bat_traditional_interp, 'LineWidth',6)
yline(12, 'LineWidth',6, 'LineStyle','--')
yline(20.5, '-', 'Set-point', 'LineWidth',6, 'LabelHorizontalAlignment','left')
yline(28, 'LineWidth',6, 'LineStyle','--')
ytop = (28)*ones(1,N_reps*2474);
ybottom = (12)*ones(1,N_reps*2474);
patch([t, flip(t)], [ybottom, ytop], [0.5, 0.5, 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.3)
axis([1, N_reps*2474, 20, 29])
legend("NN+MPC", "MPC",'Location','northwest')
xlabel("Time (s)")
ylabel(['Temperature (C' char(176) ')'])
fontsize(32, 'points')

%% heating power
figure(2)
plot(heatingPwr_nn/1000, 'LineWidth',3)
average_heatingPwr_nn = mean(heatingPwr_nn/1000)
hold on
plot(heatingPwr_traditional/1000, 'LineWidth',3)
average_heatingPwr_traditional = mean(heatingPwr_traditional/1000)

legend("NN+MPC", "MPC")
xlabel("Time (s)")
ylabel(['Power (kW)'])
fontsize(32, 'points')


%% pump power
figure(3)

plot(t,pump_power_nn_interp, 'LineWidth',3)
average_pump_power_nn = mean(pump_power_nn_interp)

hold on
plot(t,pump_power_traditional_interp, 'LineWidth',3)
average_pump_power_traditional = mean(pump_power_traditional_interp)

legend("NN+MPC", "MPC")
xlabel("Time (s)")
ylabel(['Power (kW)'])
fontsize(32, 'points')

%% pump power based on function
figure(4)
pump_power_nn_function = Pump_Power(omega_nn);
pump_power_traditional_function = Pump_Power(omega_traditional);

plot(pump_power_nn_function)
hold on
plot(pump_power_nn_interp(1:5:end))

%% Energy use based on fitted power
function P = Pump_Power(omega)
    omega_norm = omega/100;
    %constant = 50;
    %density_coolant = 1050;
    %pump_displacement = 1/(2*pi)*40/(100^3); % D parameter in simulink
    %mdot_c = density_coolant*pump_displacement.*omega;
    P = (-0.21574 +  10.0501.*omega_norm.^2 + 6.84987 .* omega_norm.^3)/1000;
end