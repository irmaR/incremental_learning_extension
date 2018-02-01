function [area]=lssvmInference(selectedtrainFea,selectedtrainLabels,testFea,testLabels,options)
gamma=1;
%fprintf('sigma %f',options.t)
features = AFEm(selectedtrainFea,options.kernel_type, options.kernel,selectedtrainFea);
try,
    [CostL3, gamma_optimal] = bay_rr(features,selectedtrainLabels,options.gamma,1);
catch,
    warning('no Bayesian optimization of the regularization parameter');
    gamma_optimal = gamma;
end
[w,b] = ridgeregress(features,selectedtrainLabels,options.gamma);
Yh0 = AFEm(selectedtrainFea,options.kernel_type, options.kernel,testFea)*w+b;
echo off;
[area,se,thresholds,oneMinusSpec,Sens]=roc(Yh0,testLabels);
end