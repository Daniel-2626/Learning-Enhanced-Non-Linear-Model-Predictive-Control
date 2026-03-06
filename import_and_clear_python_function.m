%% 
open("Adaptive_EVBatteryCoolingSystem.slx")
T_env = -10;
T_init = T_env + 273.15;
tF = 2474;
module  = py.importlib.import_module('adaptive_nn_mpc_battery');
%%
clear module
module  = py.importlib.import_module('adaptive_nn_mpc_battery');

py.importlib.reload(module)