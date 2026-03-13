%% Coefficients (taken from simulink/formula collections)
m_b = 20*2.5*4; % kg OK, double check with Prashant
c_b = 795; % J/kgK OK, double check with Prashant
c_coolant = 3500; % J/kgK OK, double check with Prashant
rho = 1050; % kg/m3, density of coolant, OK, double check with prashant
D = 1/(2*pi)*40/(100^3); % m3/rad displacement, OK, double check with Daniel; 40 cm3/rev --> 40/100^3: 
R_b = 4*20*(0.0128); %0.0028; % Ohm. Each battery pack has 20 cells, 0.0109 ohm in each
C_battery = 28 * 3600; % Ah --> As (Coloumb) to have it in SI. each battery pack has 28 Ah, 4 packs
alpha = 0.65;%0.55;
alpha_1 = 0.998321; %0.1;
alpha_2 = 0.999593; %0.319586;
hAbat = 2500;
%[0.998321, 0.999593] identified
%%
% u = w (omega, angular velocity)
u = sym('u', [2;1]);
% x = [SOC; T_b]
x = sym('x', [2;1]);
% p = [I_b; T_cool_out; T_cool_in]
p = sym('p', [3;1]);
% don't remember why but need to include time for some reason
syms t real

NTU = hAbat/(rho*D*u(1)*c_coolant + 1e-3);
T_cool_in = alpha_1*(x(1) + 1/(1-exp(-NTU))*u(2)/(rho*D*u(1)*c_coolant + 1e-3));
T_cool_out = alpha_2*((T_cool_in - x(1)) * exp(-NTU) + x(1));
f = [alpha/(m_b*c_b) * (p(1)^2*R_b - rho*D*u(1)*c_coolant*(T_cool_out - T_cool_in));
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
T_bat_true = s.T_bat;
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

%% 
%figure(2)
%plot(T_cool_in-273.15)
%plot(rho*D*c_coolant*(T_cool_out-T_cool_in)/(m_b*c_b).*ang_vel)