%% Derivative loss 
clear all;
close all;
clc;

x = linspace(0, 5, 1000);

% Define the two functions
y1 = 3*x.^2 - 0.5*exp(x) + 1*x;
y2 = 2.5*x.^2 - 0.3*exp(x) - 3*sqrt(x);

% Create the plot
figure('Position', [100, 100, 800, 600]);
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
% Plot the functions
plot(x, y1, 'b-', 'LineWidth', 4);
hold on;
plot(x, y2, 'r--', 'LineWidth', 4);

% Define positions for vertical bars
bar_positions = [3.2];  

% Color map for different bars
colors = {'k', 'g', 'm'};

for i = 1:length(bar_positions)
    x_bar = bar_positions(i);
    
    
    y1_at_bar = 3*x_bar^2 - 0.5*exp(x_bar) + 1*x_bar;
    y2_at_bar = 2.5*x_bar^2 - 0.3*exp(x_bar) - 3*sqrt(x_bar);
    
    % Draw vertical bar
    line([x_bar, x_bar], [min(y1_at_bar, y2_at_bar), max(y1_at_bar, y2_at_bar)], ...
         'Color', colors{i}, 'LineWidth', 1.5, 'LineStyle', '-');
    
    % Add delta annotation
    difference = abs(y1_at_bar - y2_at_bar);
    text(x_bar + 0.05, (y1_at_bar + y2_at_bar)/2, ...
         sprintf('$f_{\\mathrm{res}}$ = %.2f', difference), ...
         'FontSize', 36, 'BackgroundColor', 'white', ...
         'EdgeColor', colors{i}, 'Margin', 1, 'Interpreter', 'latex');
end

% Customize the plot
xlabel('x', 'FontSize', 28);
ylabel('y', 'FontSize', 28);
legend("$f$", "$f_{\mathrm{nom}}$",'Location','northwest', 'Interpreter', 'latex')
grid off;
% legend('Location', 'best', 'FontSize', 11);

% Add axis lines
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

% Set axis limits
xlim([-0.5, 5]);
ylim([-2.5, 25]);

hold off;

%% Trajectory loss 
clear all;
close all;
clc;

x = linspace(0, 5, 1000);

% Define the two functions
y1 = 3*x.^2 - 0.5*exp(x) + 1*x;
y2 = 2.5*x.^2 - 0.3*exp(x) - 3*sqrt(x);

% Create the plot
figure('Position', [100, 100, 800, 600]);
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultTextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');

% Plot the functions
plot(x, y1, 'b-', 'LineWidth', 4);
hold on;


plot(x, y2, 'r--', 'LineWidth', 4);
% --- NEW CODE: Shade the area between y1 and y2 ---
x_fill = [x, fliplr(x)];
y_fill = [y1, fliplr(y2)];
fill(x_fill, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.3, 'EdgeColor', 'none');
% --------------------------------------------------
dim = [.65 .5 .3 .3];
annotation("textbox",dim, 'EdgeColor', 'None',String="Trajectory loss")

% Define positions for vertical bars
bar_positions = [3.2];  

% Color map for different bars
colors = {'k', 'g', 'm'};
% 
% for i = 1:length(bar_positions)
%     x_bar = bar_positions(i);
% 
%     y1_at_bar = 3*x_bar^2 - 0.5*exp(x_bar) + 1*x_bar;
%     y2_at_bar = 2.5*x_bar^2 - 0.3*exp(x_bar) - 3*sqrt(x_bar);
% 
%     % Draw vertical bar
%     line([x_bar, x_bar], [min(y1_at_bar, y2_at_bar), max(y1_at_bar, y2_at_bar)], ...
%          'Color', colors{i}, 'LineWidth', 1.5, 'LineStyle', '-');
% 
%     % Add delta annotation
%     difference = abs(y1_at_bar - y2_at_bar);
%     text(x_bar + 0.05, (y1_at_bar + y2_at_bar)/2, ...
%          sprintf('$f_{\\mathrm{res}}$ = %.2f', difference), ...
%          'FontSize', 36, 'BackgroundColor', 'white', ...
%          'EdgeColor', colors{i}, 'Margin', 1, 'Interpreter', 'latex');
% end

% Customize the plot
xlabel('x', 'FontSize', 28);
ylabel('y', 'FontSize', 28);

% Corrected to single quotes to avoid the \m escape character warning
legend('$f$', '$f_{\mathrm{nom}}$', 'Location', 'northwest', 'Interpreter', 'latex')

grid off;

% Add axis lines
ax = gca;
ax.XAxisLocation = 'origin';
ax.YAxisLocation = 'origin';
set(findall(gcf, '-property', 'FontSize'), 'FontSize', 28);
set(findall(gcf, '-property', 'FontName'), 'FontName', 'Times New Roman');

% Set axis limits
xlim([-0.5, 5]);
ylim([-2.5, 25]);
hold off;