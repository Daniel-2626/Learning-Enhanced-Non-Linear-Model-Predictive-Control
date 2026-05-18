prediction = [13 15 16 17 18.5 19 20];
true = [11 13 14 14.5 14.5 17 18];

plot(prediction, '-x', 'LineWidth',4)
hold on
plot(true, '-x', 'LineWidth', 4)

x_p = [4, 4]
y_p = [14.5 17]
plot(x_p, y_p, 'Color', 'black', 'LineWidth',4)
legend("Prediction", "True")
xlabel("Time (s)")
ylabel(['Temperature (C' char(176) ')'])