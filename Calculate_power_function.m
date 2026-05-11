% calculate the the power function such that it can be used in the cost function 

% variable declarations 
Time = pump_power_logs.Time;
Power = 1000*pump_power_logs.Data; % in watts

mdotTime = mdot_logs.Time;
mdot = mdot_logs.Data; 

omegaTime = omega_logs.Time;
omega = omega_logs.Data; 

% interpolate omega to match power time series
mdot_at_Power_times = interp1(mdotTime, mdot, Time, "linear", "extrap");
omega_at_Power_times = interp1(omegaTime, omega, Time, "linear", "extrap");

% polyfit to extract function 
degree = 2;
p1 = polyfit(mdot_at_Power_times, Power, degree);
p2 = polyfit(omega_at_Power_times, Power, degree);

% create smooth curve for plotting
mdot_fit = linspace(min(mdot_at_Power_times), max(mdot_at_Power_times), 100);
omega_fit = linspace(min(omega_at_Power_times), max(omega_at_Power_times), 100);
Power_fit1 = polyval(p1, mdot_fit);
Power_fit2 = polyval(p2, omega_fit);

% visualize the result
% figure;
% plot(mdot_at_Power_times, Power, 'b.', 'DisplayName', 'Original Data');
% hold on;
% plot(mdot_fit, Power_fit1, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('Polynomial Fit (deg %d)', degree));
% xlabel('mdot (kg/s)');
% ylabel('Power (W)');
% legend('show');
% grid on;

% visualize the result
figure;
plot(mdot_at_Power_times, Power, 'b.', 'DisplayName', 'Original Data');
hold on;
plot(mdot_fit, Power_fit1, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('Polynomial Fit (deg %d)', degree));
xlabel('omega (rad/s)');
ylabel('Power (W)');
legend('show');
grid on;

% view polynomial coefficients
disp('Polynomial coefficients (highest power first):');
disp(p2);