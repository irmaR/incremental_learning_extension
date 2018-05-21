function [area]=lssvmInference(selectedtrainFea,selectedtrainLabels,testFea,testLabels,options)
gamma=1;
%We assume it's a binary problem. I needed to have classes 1 and 2.
%Hoewever, SVM assumes classes are 1 and -1. So I need to change that here.
selectedtrainLabels(selectedtrainLabels==2)=-1;
testLabels(testLabels==2)=-1;
features = AFEm(selectedtrainFea,options.kernelType, options.kernel,selectedtrainFea);
try,
    [CostL3, gamma_optimal] = bay_rr(features,selectedtrainLabels,options.gamma,1);
catch,
    warning('no Bayesian optimization of the regularization parameter');
    gamma_optimal = gamma;
end
[w,b] = ridgeregress(features,selectedtrainLabels,options.gamma);
Yh0 = AFEm(selectedtrainFea,options.kernelType, options.kernel,testFea)*w+b;
echo off;
[area,se,thresholds,oneMinusSpec,Sens]=roc(Yh0,testLabels);
end