%% Get data
EV_sys = load("EVBatteryCoolingSystem.mat");
FTP_75 = EV_sys.FTP_75_Drive_Cycle;
timeseries = getElement(FTP_75, "Current");

current = timeseries.data;
time = timeseries.Time;

%%
%%
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
plot(time,current, 'LineWidth',2)
xlabel("Time (s)")
ylabel('Current (A)', 'Interpreter', 'latex')
%axis([1, N_reps*2475, 19.5, 26])
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');
%% Interpolation
time_steps = 1:1:2474;
current_intp1 = interp1(time,current,time_steps);

%% FFT
power_fft = abs(fft(current_intp1));
plot(power_fft)

%% Validation
plot(time_steps, current_intp1, 'LineWidth',2)
hold on 
plot(time, current, 'LineStyle','--', 'LineWidth',2)
%% Saving
current_intp1_repeated = repmat(current_intp1, 1,10)
writematrix(current_intp1_repeated', "current_intp1.csv")

%% RMS
current_5s_rms = sqrt(mean(reshape(current_intp1_repeated, 5, []).^2, 1));
plot(current_5s_rms)
hold on
plot(current_intp1_repeated)

%% Saving
writematrix(current_5s_rms', "current_rms_5s.csv")
