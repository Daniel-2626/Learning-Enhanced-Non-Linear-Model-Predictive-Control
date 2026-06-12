# Master Thesis Learning-Enchanced Nonlinear Model Predictive Control
Thesis by Daniel Joseph McCauley (MPSYS) and Lech Kazimerz Kula (MPSYS), Chalmers University of Technology, 2026

## Summary
This GitHub includes a battery thermal management system simulated in Simulink, based on: https://se.mathworks.com/help/hydro/ug/ev-battery-cooling.html
The model is in the file: Adaptive_Cool_and_Heat_EVBatteryCoolingSystem
Main contributions are codes for model predictive control (MPC) formulation of battery thermal management. The goal of the theis was to solve "model mismatch" using Neural Networks. 
First, a model was developed based on physical principles and parameter fitting. 
Second, model mismatch was introduced by changing the plant, such that the model originally developed would be inaccurate.  
### Notes on usage
The thesis had mixed results for using learning-enhanced MPC in BTM systems. Thescaling in the MPC is done in an unintuitive way. Scaling was done to get inputs, states and disturbances in the same order of magnitude, instead of max scaling. The results in the thesis were obtained on this, and therefore code kept as is. When using this code, it would be good to redifine the following scales
        model.omega_scale = 100
        model.Q_heat_scale = 1000
        model.T_bat_scale = 100
        model.current_scale = 25 
so that they are max scaled. This is done properly in Vectorized_Trajectory_Loss/vectorized_trajectory_loss_adaptive_nn_mpc_battery.py but NOT tested thoroughly. 
### BTM MPC 

nominal_acados_mpc_battery_new_scaling <-- A model predictive controller for reference tracking based on a physics model 

adaptive_trajectory_loss_mpc_battery <-- An adaptive model predictive controller for reference tracking with a neural network residual added to the physics model. Neural network learns on "trajectory loss"

adaptive_derivative_loss_mpc_battery <-- An adaptive model predictive controller for reference tracking with a neural network residual added to the physics model. Neural network learns on "derivative loss"

nominal_economic_mpc_battery_new_scaling <-- An economic model predictive controller based on a physics model

adaptive_economic_trajectory_loss <-- An economic model predictive controller with a neural network residual added to the physics model. Neural network learns on "trajectory loss"

import_and_clear_python_function <-- Help code that opens the model and imports the python programs for use within Simulink
### Offline Training
Using an untrained network in MPC degrades performance. Therefore there is code for offline training the neural networks in the folder Offline_Training/, together with pretrained networks and data collected on matched/mismatched systems

### Vectorized Trajectory Loss
The implemenation of trajectory loss had long training. To shorten training times, a vectorized formulation was implemented, available in Vectorized_Trajectory_Loss/.
### Cascaded Tanks Benchmark
For a quick introduction to the workings of the learning-enhanced nonlinear model predictive control code for reference tracking of two Cascaded Tanks is included in the folder Cascaded_Tanks/
Here there is code for MPC based on a physics model (nominal_mpc_cascaded_tanks), adaptive MPC using trajectory loss (traj_adaptive_nn_mpc_cascaded_tanks) and adaptive MPC using derivative loss (deriv_adaptive_nn_mpc_cascaded_tanks)
### Other code
The rest of the code is not strictly necessary to run the BTM/Cascaded Tanks MPC. However, they include plotting and analysis scripts. There is no guarantee that they work in their current states, some files might have to be moved due to changes in folder structure. 
