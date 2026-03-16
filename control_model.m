%% Coefficients (taken from simulink/formula collections)
% Parameters
m_battery = 20*2.5*4;
c_battery = 795;
c_coolant = 3500;
density_coolant = 1050;

pump_displacement = 1/(2*pi)*40/(100^3); % D parameter in simulink
R_battery = 4*20*0.0128; % Battery resistance
C_battery = 28*3600; % in coloumb
hA_bat = 2500;

T_env = 0 + 273.15;

%[0.637968, 1, 0.980682, 0.981036]
alpha_0 = 0.637968; %0.65
alpha_1 = 1; % 0.99 %0.998427 #0.99
alpha_2 = 0.980682; % #0.999686 #.97
alpha_3 = 0.981036; %
gamma = 7.59853;
%%
% u = w (omega, angular velocity)
u = sym('u', [2;1]);
% x = [SOC; T_b]
x = sym('x', [2;1]);
% p = [I_b; T_cool_out; T_cool_in]
p = sym('p');
% don't remember why but need to include time for some reason
syms t real
mdot_c = density_coolant*pump_displacement*u(1);
NTU = alpha_3*hA_bat/(mdot_c*c_coolant + 1e-3);
T_cool_in = (x(1) + alpha_1*1/(1-exp(-NTU))*u(2)/(mdot_c*c_coolant + 1e-3));
T_cool_out = ((T_cool_in - x(1)) * alpha_2*exp(-NTU) + x(1));
Q_cool = mdot_c * c_coolant * (T_cool_out - T_cool_in);
f = [alpha_0/(m_battery*c_battery) * (p(1)^2*R_battery - Q_cool + gamma*(T_env-x(1)));
    -p(1)/C_battery];
matlabFunction(f, 'File', 'dynamics', 'Vars', {t, x, u, p})

%% load parameters
filename="MPC_heating_0.mat";
load(filename)
dt = s.dt; 
tF = s.tF;
current = s.current;
%T_cool_out = s.T_cool_out;
%T_cool_in = s.T_cool_in;
%ang_vel = s.ang_vel;

SOC_true = s.SOC;
T_bat_true = s.T_bat + 273.15;
omega = s.omega;
Q_heat = s.Q;

%%
x0 =[T_bat_true(1); SOC_true(1)];

[dimension , ~] = size(x0) ;
N = tF/dt;

xRK4 = zeros(dimension,N+1);
xRK4(:,1) = x0; 
parameters = current;
inputs = [omega; Q_heat];

tRK4 = zeros(1,N+1);
tRK4(1) = 0;

c1 = 0;
c2 = 1/2;
c3 = 1/2;
c4 = 1;

a21 = 1/2;
a32 = 1/2;
a43 = 1;

b1 = 1/6;
b2 = 1/3;
b3 = 1/3;
b4 = 1/6;

for k=1:N
    %current_hold = parameters(ceil(k/10));
    K1 = dynamics(tRK4(k),xRK4(:,k), inputs(:,k), current(k)); 
    K2 = dynamics(tRK4(k),xRK4(:,k)+ dt*a21*K1,inputs(:,k), current(k));
    K3 = dynamics(tRK4(k),xRK4(:,k)+dt*a32*K2, inputs(:,k), current(k));
    K4 = dynamics(tRK4(k),xRK4(:,k)+dt*a43*K3, inputs(:,k), current(k));
    xRK4(:,k+1) = xRK4(:,k) + dt*(b1*K1 + b2*K2 + b3*K3 + b4*K4);

    tRK4(k+1) = tRK4(k) + dt;
end

%%
figure(1)
plot(tRK4, xRK4(1,:), 'LineWidth',2)
hold on
plot(tRK4, T_bat_true, 'LineWidth', 2, 'LineStyle','--')
legend(["model", "true"])

%% Finding steady state
omega_ss = 100;
T_bat_ss = 20.5 + 273.15;
I_ss = 0;

mdot_c_ss = density_coolant*pump_displacement*omega_ss;
NTU_ss = alpha_3*hA_bat/(mdot_c_ss*c_coolant + 1e-3);
T_cool_in_ss = (T_bat_ss + alpha_1*1/(1-exp(-NTU_ss))*u(2)/(mdot_c_ss*c_coolant + 1e-3));
T_cool_out_ss = ((T_cool_in_ss - T_bat_ss) * alpha_2*exp(-NTU_ss) + T_bat_ss);
Q_cool_ss = mdot_c_ss * c_coolant * (T_cool_out_ss - T_cool_in_ss);
f_ss = alpha_0/(m_battery*c_battery) * (I_ss^2*R_battery - Q_cool_ss + gamma*(T_env-T_bat_ss));
q_heat_ss = double(solve(f_ss == 0, u(2)));
