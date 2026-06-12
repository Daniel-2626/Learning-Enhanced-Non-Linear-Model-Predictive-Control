%% Replicate speed data

speed_data = FTP_75_Drive_Cycle{1};
tF = speed_data.TimeInfo.End;
N_reps = 13;
time_speed = speed_data.Time;
[idx_final, ~] = size(time_speed); 
time_speed_rep = repmat(time_speed, [N_reps,1]);

for n = 1:N_reps
    time_speed_rep(n*idx_final+1:end) = time_speed_rep(n*idx_final+1:end) + tF;
end    
speed = speed_data.Data;
speed_rep = repmat(speed, [N_reps,1]);

%% Replicate current data
current_data = FTP_75_Drive_Cycle{2};
tF = current_data.TimeInfo.End;
N_reps = 13;
time_current = current_data.Time;
[idx_final, ~] = size(time_current); 
time_current_rep = repmat(time_current, [N_reps,1]);

for n = 1:N_reps
    time_current_rep(n*idx_final+1:end) = time_current_rep(n*idx_final+1:end) + tF;
end    
current = current_data.Data;
current_rep = repmat(current, [N_reps,1]);

%% Copy and replace
speed_ts = timeseries(speed_rep, time_speed_rep);
current_ts = timeseries(current_rep, time_current_rep);
FTP_75_Drive_Cycle_Extended = table(speed_ts, current_ts, 'VariableNames',["Speed", "Current"]);

save("EV_Extended.mat", "FTP_75_Drive_Cycle_Extended")

%%
figure(1)
plot(speed_ts)
figure(2)
plot(current_ts)