%% Example
%% Load diabetes data
clear;
load diabetes;
X = diabetes_X;
y = diabetes_Y;

n = size(X,1);

%% Augment the design matrix with large number of additional variables
Xn = [X, randn(n, 4e3)];
p  = size(Xn,2);

fprintf('*****************************************************************************\n');
fprintf('* fastridge example using diabetes data (augmented with extra noise variables):\n');
fprintf('  - n = %d\n', n);
fprintf('  - p = %d\n', p);
fprintf('\n');

%% Produce a regularisation path
fprintf('* Producing a regularisation path of 100 betas using fastridge ');
tic;
[beta, b0, tau2, DOF, lambda, score] = fastridge(Xn, y, 'path', 1e2);
fprintf('(%.2fs)\n', toc);

[~,I] = min(score);
fprintf('  - DOF of best model    = %.2f\n', DOF(I));
fprintf('  - lambda of best model = %.2f\n', lambda(I));

displaypath(beta, DOF, score);

fprintf('\n');

%% Fit model with specified DOF
fprintf('* Fitting model with %.2f DOF ', DOF(I));
tic;
[b, ~, ~, ~, ~, ~] = fastridge(Xn, y, 'DOF', DOF(I));
fprintf('(%.2fs)\n', toc);
fprintf('  - L2 norm between this model and best model from path = %.2f\n', norm(b-beta(:,I)));

fprintf('\n');

%% Fit model with specified lambda
fprintf('* Fitting model with lambda=%.2f ', lambda(I));
tic;
[b, ~, ~, ~, ~, ~] = fastridge(Xn, y, 'lambda', lambda(I));
fprintf('(%.2fs)\n', toc);
fprintf('  - L2 norm between this model and best model from path = %.2f\n', norm(b-beta(:,I)));

fprintf('\n');

%% Search for best model using MML criterion
fprintf('* Fitting model using MML (searching for lambda) ');
tic;
[b_srch, ~, ~, DOF_srch, lambda_srch] = fastridge(Xn, y);
fprintf('(%.2fs)\n', toc);
fprintf('  - DOF of this model    = %.2f\n', DOF_srch);
fprintf('  - lambda of this model = %.2f\n', lambda_srch);
fprintf('  - L2 norm between this model and best model from path = %.2f\n', norm(b_srch-beta(:,I)));

fprintf('\n');

%% Search for best model using AICc criterion
fprintf('* Fitting model using AICc (searching for lambda) ');
tic;
[b_srch, ~, ~, DOF_srch, lambda_srch] = fastridge(Xn, y, 'criterion', 'aicc');
fprintf('(%.2fs)\n', toc);
fprintf('  - DOF of this model    = %.2f\n', DOF_srch);
fprintf('  - lambda of this model = %.2f\n', lambda_srch);
fprintf('  - L2 norm between this model and best model from MML path = %.2f\n', norm(b_srch-beta(:,I)));

fprintf('\n');

%% Finally fit a model using the standard ridge
fprintf('* Fitting model with lambda=%.2f using MATLAB Ridge ', lambda(I));
tic;
b_matlab = ridge(y,Xn,lambda(I),0);
fprintf('(%.2fs)\n', toc);
fprintf('  - L2 norm between this model and best model from path = %.2f\n', norm(b_matlab(2:end)-beta(:,I)));