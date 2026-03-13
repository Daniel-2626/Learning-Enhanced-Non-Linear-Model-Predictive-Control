logged_data = load("MPC_heating_t_env_0.mat");
outputs = logged_data.data;

test_name = "MPC_heating_0";

%T_cool_out = getElement(outputs, 'T_cool_out').Values.Data;
%T_cool_in = getElement(outputs, 'T_cool_in').Values.Data;
current = getElement(outputs, 'current').Values.Data;
SOC = getElement(outputs, 'SOC').Values.Data;
T_bat = getElement(outputs, 'Pack3').Values.Data;
Q = getElement(outputs, 'input_q_heat').Values.Data;
omega = getElement(outputs, 'input_omega').Values.Data;
time = getElement(outputs, "time").Values.Data;



%% Interpolation
tF = time(end);
dt = 1; %tF/10^(ceil(log10(length(tout))));
t = (0:dt:tF);
current_intp1 = interp1(time, current, t);
%T_cool_out_intp1 = interp1(timesteps, T_cool_out, t);
%T_cool_in_intp1 = interp1(timesteps, T_cool_in, t);
SOC_intp1 = interp1(time, SOC, t);

T_bat_intp1 = interp1(time, T_bat, t);

Q_intp1 = Q'; %interp1(timesteps, Q, t);
omega_intp1 = omega';
%% Save data
s = struct;
s.time = time;
s.dt = dt;
s.tF = tF;
s.current = current_intp1;
%s.T_cool_out = T_cool_out_intp1;
%s.T_cool_in = T_cool_in_intp1;
%s.ang_vel = ang_vel_intp1;
s.SOC = SOC_intp1;
s.T_bat = T_bat_intp1;
s.Q = Q_intp1;
s.omega = omega_intp1;

filename=test_name+".mat";

save(filename, 's')

%%
plot(t, SOC_intp1)
hold on 
plot(time, SOC, '--')
%plot(T_cool_in, '--')