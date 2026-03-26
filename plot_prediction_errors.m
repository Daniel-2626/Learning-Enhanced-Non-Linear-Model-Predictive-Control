logged_data_nn = load("pred_error_nn.mat");
outputs_nn = logged_data_nn.data;
logged_data_no_nn = load("pred_error_no_nn.mat");
outputs_no_nn = logged_data_no_nn.data;

pred_error_nn = abs(getElement(outputs_nn, "pred_error").Values.Data(1:end));
pred_error_no_nn = abs(getElement(outputs_no_nn, "pred_error").Values.Data(1:end));



%% Plotting

[T, ~] = size(pred_error_nn);
t = 0:30:30*(T-1)';
plot(t, pred_error_no_nn, 'LineWidth',2)
hold on
plot(t, pred_error_nn, 'LineWidth',2, 'LineStyle','--')
yline(0)
xline(300,'-',{'NN on after this point'}, 'LineWidth', 2);
legend(["Nominal", "NN"])
title("NN vs Nominal with timestep of 30 s")
