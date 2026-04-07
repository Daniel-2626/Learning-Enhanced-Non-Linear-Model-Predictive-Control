logged_data_nn = load("pred_error_nn_specific_heat.mat");
outputs_nn = logged_data_nn.data;
logged_data_no_nn = load("pred_error_nominal_specific_heat.mat");
outputs_no_nn = logged_data_no_nn.data;

pred_error_nn = abs(getElement(outputs_nn, "pred_error").Values.Data(1:end));
pred_error_nominal = abs(getElement(outputs_no_nn, "pred_error").Values.Data(1:end));

omega_nn = getElement(outputs_nn, "input_omega").Values.Data(1:end);
omega_nominal = getElement(outputs_no_nn, "input_omega").Values.Data(1:end);

Q_heat_nn = getElement(outputs_nn, "input_q_heat").Values.Data(1:end);
Q_heat_nominal = getElement(outputs_no_nn, "input_q_heat").Values.Data(1:end);



%% Plotting
figure(1)
[T, ~] = size(pred_error_nn);
t = 0:5:5*(T-1)';
plot(t, pred_error_nn, 'LineWidth',2, 'LineStyle','-')

hold on
plot(t, pred_error_nominal, 'LineWidth',2, 'LineStyle','--')

yline(0)
xline(180,'-',{'NN on after this point'}, 'LineWidth', 2, 'LabelHorizontalAlignment', 'left');
legend(["NN", "Nominal"])
title("NN vs Nominal with timestep of 30 s")

%%
figure(2)
plot(Q_heat_nn)
hold on
plot(Q_heat_nominal)
legend("NN", "NOMINAL")