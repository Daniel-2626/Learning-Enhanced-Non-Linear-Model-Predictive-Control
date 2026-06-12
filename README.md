# Master Thesis Learning-Enchanced Nonlinear Model Predictive Control
Thesis by Daniel Joseph McCauley (MPSYS) and Lech Kazimerz Kula (MPSYS), Chalmers University of Technology, 2026

## Summary
This GitHub includes a battery thermal management system simulated in Simulink, based on: https://se.mathworks.com/help/hydro/ug/ev-battery-cooling.html
The model is in the file: Adaptive_Cool_and_Heat_EVBatteryCoolingSystem
Main contributions are codes for model predictive control (MPC) formulation of battery thermal management. 
A quick description of the codes are:
nominal_acados_mpc_battery_new_scaling <-- A model predictive controller for reference tracking based on a physics model 
adaptive_trajectory_loss_mpc_battery <-- An adaptive model predictive controller for reference tracking with a neural network residual added to the physics model. Neural network learns on "trajectory loss"
adaptive_derivative_loss_mpc_battery <-- An adaptive model predictive controller for reference tracking with a neural network residual added to the physics model. Neural network learns on "derivative loss"
nominal_economic_mpc_battery_new_scaling <-- An economic model predictive controller based on a physics model
adaptive_economic_trajectory_loss <-- An economic model predictive controller with a neural network residual added to the physics model. Neural network learns on "trajectory loss"
import_and_clear_python_function <-- Help code that opens the model and imports the python programs for use within Simulink

### Cascaded Tanks Benchmark
For a quick introduction to the workings of the learning-enhanced nonlinear model predictive control code for reference tracking of two Cascaded Tanks is included in the folder Cascaded_Tanks/
