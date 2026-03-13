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