# Learning-Enhanced Nonlinear Model Predictive Control

> **Master's Thesis | Chalmers University of Technology, 2026**
> **Authors:** Daniel Joseph McCauley (MPSYS) & Lech Kazimerz Kula (MPSYS)

## Overview

This repository contains a Battery Thermal Management (BTM) system simulated in Simulink, based on the [MathWorks EV Battery Cooling System model](https://se.mathworks.com/help/hydro/ug/ev-battery-cooling.html).

* **Core Model:** `Adaptive_Cool_and_Heat_EVBatteryCoolingSystem`
* **Main Contribution:** Code for the Model Predictive Control (MPC) formulation of battery thermal management.

The primary goal of this thesis was to solve "model mismatch" using Neural Networks. The methodology followed two main steps:

1. A baseline model was developed based on physical principles and parameter fitting.
2. Model mismatch was intentionally introduced by altering the plant, rendering the originally developed model inaccurate so the learning-enhanced controller could be tested.

---

## Important Notes on Usage

The thesis yielded mixed results regarding the application of learning-enhanced MPC in BTM systems.

**Scaling Considerations**
The scaling in the MPC is implemented in an unintuitive way. Rather than using max scaling, it was designed to bring inputs, states, and disturbances into the same order of magnitude. Because the thesis results were obtained using this configuration, the code has been kept as-is to preserve reproducibility.

When using or adapting this code, it is highly recommended to redefine the following scales to achieve proper max scaling:

```python
model.omega_scale = 100
model.Q_heat_scale = 1000
model.T_bat_scale = 100
model.current_scale = 25 

```

*Note: Proper max scaling is implemented in `Vectorized_Trajectory_Loss/vectorized_trajectory_loss_adaptive_nn_mpc_battery.py`, but it has **not** been thoroughly tested.*

---

## Repository Structure & Modules

### Battery Thermal Management (BTM) MPC

| Script / Controller | Description |
| --- | --- |
| `nominal_acados_mpc_battery_new_scaling` | An MPC for reference tracking based purely on a physics model. |
| `adaptive_trajectory_loss_mpc_battery` | An adaptive MPC for reference tracking. A neural network residual is added to the physics model and learns via "trajectory loss." |
| `adaptive_derivative_loss_mpc_battery` | An adaptive MPC for reference tracking. A neural network residual is added to the physics model and learns via "derivative loss." |
| `nominal_economic_mpc_battery_new_scaling` | An economic MPC based strictly on a physics model. |
| `adaptive_economic_trajectory_loss` | An economic MPC with a neural network residual added to the physics model, learning via "trajectory loss." |
| `import_and_clear_python_function` | Helper code that opens the model and imports the required Python programs for use within Simulink. |

### Offline Training

Using an untrained network in the MPC degrades performance. The `Offline_Training/` folder contains:

* Code for offline training of the neural networks.
* Pre-trained networks.
* Data collected on matched and mismatched systems.

### Vectorized Trajectory Loss

The original implementation of trajectory loss resulted in long training times. To accelerate this process, a vectorized formulation was implemented and is available in the `Vectorized_Trajectory_Loss/` directory.

### Cascaded Tanks Benchmark

For a quick introduction to the inner workings of the learning-enhanced nonlinear MPC code, a reference tracking benchmark for two cascaded tanks is included in the `Cascaded_Tanks/` folder.

| Script | Description |
| --- | --- |
| `nominal_mpc_cascaded_tanks` | Baseline MPC based on a physics model. |
| `traj_adaptive_nn_mpc_cascaded_tanks` | Adaptive MPC utilizing trajectory loss. |
| `deriv_adaptive_nn_mpc_cascaded_tanks` | Adaptive MPC utilizing derivative loss. |

### Other Code & Analysis

The remainder of the repository contains plotting and analysis scripts.

> **Note:** These scripts are not strictly necessary to run the BTM or Cascaded Tanks MPC code. There is no guarantee that they work flawlessly in their current state; some files may require path adjustments due to changes in the folder structure.
