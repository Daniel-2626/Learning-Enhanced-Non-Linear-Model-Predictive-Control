%% Get data
logged_data_traditional = load("power_traditional.mat");
outputs_traditional = logged_data_traditional.data;

time_traditional = getElement(outputs_traditional, "time").Values.Data;
pump_power_traditional = getElement(outputs_traditional, "pump_power").Values.Data;
heatingPwr_traditional = getElement(outputs_traditional, "heatingPwr").Values.Data;
T_bat_traditional = getElement(outputs_traditional, "Pack3").Values.Data;

%% Get data
logged_data_mpc = load("power_mpc.mat");
outputs_mpc = logged_data_mpc.data;

time_mpc = getElement(outputs_mpc, "time").Values.Data;
pump_power_mpc = getElement(outputs_mpc, "pump_power").Values.Data;
heatingPwr_mpc = getElement(outputs_mpc, "heatingPwr").Values.Data;
T_bat_mpc = getElement(outputs_mpc, "Pack3").Values.Data;


%% Interpolation
dt = 1;
t = 1:dt:2474;
pump_power_traditional_interp = interp1(time_traditional, pump_power_traditional, t);
heating_power_traditional_interp = interp1(time_traditional, heatingPwr_traditional, t);
T_bat_traditional_interp = interp1(time_traditional, T_bat_traditional, t);

pump_power_mpc_interp = interp1(time_mpc, pump_power_mpc, t);
time_mpc_heat_pwr = 0:30:2474;
heating_power_mpc_interp = interp1(time_mpc, heatingPwr_mpc, t,'previous', 'extrap');
%heating_power_mpc_interp(isnan(heating_power_mpc_interp))=0;
T_bat_mpc_interp = interp1(time_mpc, T_bat_mpc, t);

plot(time_mpc,heatingPwr_mpc)
hold on
plot(t,heating_power_mpc_interp, 'LineStyle','--')
%% Riemman sum
hour_in_sec = 1/3600;
heating_energy_traditional = hour_in_sec*sum(heating_power_traditional_interp)/1000; % Heating power in Watts, convert to kW
pump_energy_traditional = hour_in_sec*sum(pump_power_traditional_interp);
energy_traditional = heating_energy_traditional + pump_energy_traditional;
heating_energy_mpc = hour_in_sec*sum(heating_power_mpc_interp)/1000; % heating power set every 5 s
pump_energy_mpc = hour_in_sec*sum(pump_power_mpc_interp);
energy_mpc = heating_energy_mpc + pump_energy_mpc;

%%
plot(t,T_bat_mpc_interp, 'LineWidth',6)
hold on
plot(t,T_bat_traditional_interp, 'LineWidth',6)
yline(20.5-1, 'LineWidth',6, 'LineStyle','--')
yline(20.5, '-', 'Set-point', 'LineWidth',6, 'LabelHorizontalAlignment','left')
yline(20.5+1, 'LineWidth',6, 'LineStyle','--')
ytop = (20.5+1)*ones(1,2474);
ybottom = (20.5-1)*ones(1,2474);
patch([t, flip(t)], [ybottom, ytop], [0.5, 0.5, 0.5], 'EdgeColor', 'none', 'FaceAlpha', 0.3)
axis([1000, 2474, 17, 23])
legend("MPC", "Traditional")
xlabel("Time (s)")
ylabel(['Temperature (C' char(176) ')'])