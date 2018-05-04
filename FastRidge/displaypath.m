function displaypath(b, DOF, score)

%% Create a new figure
figure;
clf;
hold on;
grid on;
box on;

title('Ridge Regression Path');

% Display the path
plot(DOF, b);
xlim([0, max(DOF)]);
xlabel('Degrees of Freedom');
ylabel('Regression Coefficients');

% Show the minimum criterion score model
[~,I] = min(score);
yl = ylim;
plot([DOF(I), DOF(I)], [yl(1), yl(2)], '--');

%% Create a new figure
figure;
clf;
hold on;
grid on;
box on;

title('Ridge Regression Criterion Score');

% Display the path
plot(DOF, score);
xlim([0, max(DOF)]);
xlabel('Degrees of Freedom');
ylabel('Criterion Score');

% Show the minimum criterion score model
[~,I] = min(score);
yl = ylim;
plot([DOF(I), DOF(I)], [yl(1), yl(2)], '--');

