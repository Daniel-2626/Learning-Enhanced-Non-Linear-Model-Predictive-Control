%% 
% rmdir("_l4c_generated/",'s')
% rmdir("c_generated_code/", 's')
open("Adaptive_EVBatteryCoolingSystem.slx")
T_env = -10;
T_init = T_env + 273.15;
tF = 2474;
module  = py.importlib.import_module('adaptive_nn_mpc_battery');
%%
clear module
% rmdir("_l4c_generated/", 's')
% rmdir("c_generated_code/", 's')
% module  = py.importlib.import_module('adaptive_nn_mpc_battery');
module  = py.importlib.import_module('nominal_acados_mpc_battery');
py.importlib.reload(module)