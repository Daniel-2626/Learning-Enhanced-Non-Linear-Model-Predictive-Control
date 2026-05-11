%% Get data
data = load("MPC_heating_parameter_estimmat.mat");
dataset = data.data;
T_bat = getElement(dataset, "Pack3").Values.Data + 273.15;
input_omega = getElement(dataset, "input_omega").Values.Data;
input_q_heat = getElement(dataset, "input_q_heat").Values.Data;
current = getElement(dataset, "current").Values.Data;
time = getElement(dataset, "time").Values.Data;
%current = timeseries.data;
%time = timeseries.Time;

%% Interpolation
time_steps = 1:1:2474;
current_intp1 = interp1(time,current,time_steps)';
T_bat_intp1 = interp1(time,T_bat,time_steps)';
input_omega_intp1 = input_omega(1:end-1);
input_q_heat_intp1 = input_q_heat(1:end-1);

%% Validation
plot(time_steps, T_bat_intp1, 'LineWidth',2)
hold on 
plot(time, T_bat, 'LineStyle','--', 'LineWidth',2)
%% Saving
writematrix([T_bat_intp1, input_omega_intp1, input_q_heat_intp1, current_intp1], "true_data_intp1.csv")

%% Get Power 
data = load("Simulation_Data/power_intp_data.mat");
dataset = data.data;
input_omega = getElement(dataset, "input_omega").Values.Data(1:end-1);
pump_power = 1000*getElement(dataset, "pump_power").Values.Data; % In watts

time = getElement(dataset, "time").Values.Data;
%current = timeseries.data;
%time = timeseries.Time;

%% Interpolation
time_steps = 1:5:300;
pump_power_intp1 = interp1(time,pump_power,time_steps)';

%% Validation
plot(time_steps(3:end), pump_power_intp1(3:end), 'LineWidth',2)
hold on 
plot(time, pump_power, 'LineStyle','--', 'LineWidth',2)

%% Relation (shorten data to remove transient)
plot(input_omega(3:end))
hold on
plot(pump_power_intp1(3:end))

%% Removing transient in dataset
input_omega = input_omega(3:end);
pump_power_intp1 = pump_power_intp1(3:end);
%% Saving
writematrix([pump_power_intp1, input_omega], "power_data_intp1.csv")