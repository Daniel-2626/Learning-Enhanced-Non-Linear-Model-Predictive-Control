%% 
%rmdir("_l4c_generated/",'s')
rmdir("c_generated_code/", 's') % Remember to uncomment when debugging
open("Adaptive_EVBatteryCoolingSystem.slx")
T_env = 0;
T_init = T_env + 273.15;
tF = 2474;
t_sim = 2474;
adaptive  = py.importlib.import_module('adaptive_nn_mpc_battery');
nominal = py.importlib.import_module('nominal_acados_mpc_battery');
lms = py.importlib.import_module('lms_acados_mpc_battery');

%%
clear adaptive, clear nominal, clear lms
%rmdir("_l4c_generated/", 's')
rmdir("c_generated_code/", 's')
adaptive  = py.importlib.import_module('adaptive_nn_mpc_battery');
nominal = py.importlib.import_module('nominal_acados_mpc_battery');
lms = py.importlib.import_module('lms_acados_mpc_battery');

nominal_d = py.importlib.import_module('nominal_acados_Daniels_version');
py.importlib.reload(adaptive)
py.importlib.reload(nominal)
py.importlib.reload(nominal_d)
py.importlib.reload(lms)
