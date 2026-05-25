%% 
rmdir("_l4c_generated/",'s')
rmdir("c_generated_code/", 's') % Remember to uncomment when debugging
open("Adaptive_Cool_and_Heat_EVBatteryCoolingSystem.slx")
%open("Traditional_EVBatteryCoolingSystem.slx")
T_env = 36;
T_init = T_env + 273.15;
tF = 3*2474;
t_sim = tF;
%adaptive  = py.importlib.import_module('adaptive_nn_mpc_battery');
adaptive_scaling  = py.importlib.import_module('adaptive_nn_mpc_battery_new_scaling');
adaptive_new_loss  = py.importlib.import_module('adaptive_nn_mpc_battery_new_loss');
nominal = py.importlib.import_module('nominal_acados_mpc_battery_new_scaling');

%lms = py.importlib.import_module('lms_acados_mpc_battery');
economic = py.importlib.import_module('nominal_acados_mpc_battery_powercost_new_scaling')
adaptive_economic_new_scaling = py.importlib.import_module('adaptive_nn_mpc_battery_new_scaling');
adaptive_economic_new_loss = py.importlib.import_module('adaptive_thermal_management_new_scaling');

%%
clear adaptive, clear nominal, clear lms, clear economic, clear adaptive_economic_new_scaling, clear adaptive_scaling, clear adaptive_new_loss, clear adaptive_economic_new_loss
%rmdir("_l4c_generated/", 's')
rmdir("c_generated_code/", 's')
%adaptive  = py.importlib.import_module('adaptive_nn_mpc_battery');

adaptive_scaling  = py.importlib.import_module('adaptive_nn_mpc_battery_new_scaling');
adaptive_new_loss  = py.importlib.import_module('adaptive_nn_mpc_battery_new_loss');
nominal = py.importlib.import_module('nominal_acados_mpc_battery_new_scaling');
%lms = py.importlib.import_module('lms_acados_mpc_battery');

economic = py.importlib.import_module('nominal_acados_mpc_battery_powercost_new_scaling');
adaptive_economic_new_scaling = py.importlib.import_module('adaptive_thermal_management_new_scaling');
adaptive_economic_new_loss = py.importlib.import_module('adaptive_thermal_management_new_loss');

%py.importlib.reload(adaptive)
py.importlib.reload(adaptive_scaling)
py.importlib.reload(nominal)
%py.importlib.reload(lms)
py.importlib.reload(economic)
py.importlib.reload(adaptive_economic_new_scaling)
py.importlib.reload(adaptive_new_loss)
py.importlib.reload(adaptive_economic_new_loss)