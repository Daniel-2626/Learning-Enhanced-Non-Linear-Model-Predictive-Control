%% 
%rmdir("_l4c_generated/",'s')
%rmdir("c_generated_code/", 's')
open("Adaptive_EVBatteryCoolingSystem.slx")
T_env = -10;
T_init = T_env + 273.15;
tF = 2474;
adaptive  = py.importlib.import_module('adaptive_nn_mpc_battery');
nominal = py.importlib.import_module('nominal_acados_mpc_battery');
%%
clear adaptive, clear nominal
%rmdir("_l4c_generated/", 's')
%rmdir("c_generated_code/", 's')
adaptive  = py.importlib.import_module('adaptive_nn_mpc_battery');
nominal = py.importlib.import_module('nominal_acados_mpc_battery');
py.importlib.reload(adaptive)
py.importlib.reload(nominal)
cost = 1999.2890938141577;