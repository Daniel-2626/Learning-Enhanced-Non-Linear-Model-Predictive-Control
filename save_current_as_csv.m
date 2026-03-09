%% Get data
EV_sys = load("EVBatteryCoolingSystem.mat");
FTP_75 = EV_sys.FTP_75_Drive_Cycle;
timeseries = getElement(FTP_75, "Current");

current = timeseries.data;
time = timeseries.Time;

%% Interpolation
time_steps = 1:1:2474;
current_intp1 = interp1(time,current,time_steps);

%% Validation
plot(time_steps, current_intp1, 'LineWidth',2)
hold on 
plot(time, current, 'LineStyle','--', 'LineWidth',2)
%% Saving
writematrix(current_intp1', "current_intp1.csv")