T_f = 10;
N = 200;
f = @(x) -0.05*x.^2 + 28 + cos(x);
x = linspace(0,T_f, N);
y = f(x);

plot(x(1:140),y(1:140), 'LineWidth',6, 'Color', '#1171BE')
hold on
plot(x(141:end),y(141:end), 'LineWidth',6, 'LineStyle',':', 'Color', '#1171BE')

%yline(22.5, 'LineWidth',6, 'LineStyle','--')
%yline(28, 'LineWidth',6, 'LineStyle','--')
ytop = (28)*ones(1,N);
ybottom = (22)*ones(1,N);
patch([x, flip(x)], [ybottom, ytop], [0.5, 0.5, 0.5], 'FaceColor', '#50C878', 'EdgeColor', 'none', 'FaceAlpha', 0.3)
axis([0, T_f, 20, 31])
xl = xline(x(141), '-',{'Current time'}, 'LineWidth',6, 'LabelOrientation', 'horizontal', 'LabelVerticalAlignment', 'bottom', 'LabelHorizontalAlignment', 'right');
ax = gca;

% Data coordinates
x1 = 1.6;  y1 = 29.6;
x2 = 0.5;  y2 = 29;

% Convert to normalized figure coordinates
ax_pos = ax.Position;

x_lim = ax.XLim;
y_lim = ax.YLim;

x_an = ax_pos(1) + ( [x1 x2] - x_lim(1) ) / diff(x_lim) * ax_pos(3);
y_an = ax_pos(2) + ( [y1 y2] - y_lim(1) ) / diff(y_lim) * ax_pos(4);

annotation('textarrow', x_an, y_an, ...
    'String', 'Input applied to force into target region', ...
    'LineWidth', 3);
annotation("textbox", 'EdgeColor', 'None',String="Target region")
legend("Trajectory", "Planned trajectory", 'Location','northeast')
xlabel("Time")
ylabel('State')
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Arial');