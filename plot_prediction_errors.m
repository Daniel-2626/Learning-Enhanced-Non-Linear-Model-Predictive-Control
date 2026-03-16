logged_data_nominal = load("pred_error_nominal.mat");
outputs_nominal = logged_data_nominal.data;
logged_data_lms = load("pred_error_lms.mat");
outputs_lms = logged_data_lms.data;

pred_error_nominal = outputs_nominal.Data(11:end);
pred_error_lms = getElement(outputs_lms, "pred_err").Values.Data(11:end);



%% Plotting
tF = 2474; %time(end);
dt = 1; %tF/10^(ceil(log10(length(tout))));
t = (10:dt:tF)';

plot(t, pred_error_nominal, 'LineWidth',2)
hold on
plot(t, pred_error_lms, 'LineWidth',2)
xline(1700-10,'-',{'Inputs start to','decrease'}, 'LineWidth', 2);
legend(["Nominal", "LMS"])