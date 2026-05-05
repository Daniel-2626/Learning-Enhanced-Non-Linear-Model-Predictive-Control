%% 
%rmdir("_l4c_generated/",'s')
%rmdir("c_generated_code/", 's') % Remember to uncomment when debugging
%open("Adaptive_Cool_and_Heat_EVBatteryCoolingSystem.slx")
%open("Traditional_EVBatteryCoolingSystem.slx")
T_env = 40;
T_init = T_env + 273.15;
tF = 2*2474;
t_sim = tF;
adaptive  = py.importlib.import_module('adaptive_nn_mpc_battery');
nominal = py.importlib.import_module('nominal_acados_mpc_battery');
lms = py.importlib.import_module('lms_acados_mpc_battery');
economic = py.importlib.import_module('nominal_acados_mpc_battery_powercost')
adaptive_economic = py.importlib.import_module('adaptive_thermal_management');

%%
clear adaptive, clear nominal, clear lms, clear economic, clear adaptive_economic
%rmdir("_l4c_generated/", 's')
rmdir("c_generated_code/", 's')
adaptive  = py.importlib.import_module('adaptive_nn_mpc_battery');
nominal = py.importlib.import_module('nominal_acados_mpc_battery');
lms = py.importlib.import_module('lms_acados_mpc_battery');
economic = py.importlib.import_module('nominal_acados_mpc_battery_powercost');
adaptive_economic = py.importlib.import_module('adaptive_thermal_management');
py.importlib.reload(adaptive)
py.importlib.reload(nominal)
py.importlib.reload(lms)
py.importlib.reload(economic)
py.importlib.reload(adaptive_economic)