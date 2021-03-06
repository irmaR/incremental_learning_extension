function [area]=srdaInference(~,trainFea,trainLabels,testFea,testLabels,options)
%if we only have one class, return area=0
if length(unique(trainLabels))==1
    fprintf('only one label selected!\n')
    unique(trainLabels)
    fprintf('\n')
    area=NaN;
    return
end
options.ReguType = 'Ridge';
options.gnd = trainLabels;
[eigvector, ~] = SR(options,trainLabels,trainFea);
Yhat = testFea*eigvector; %projection
if sum(isnan(Yhat))~=0
    area=0;
else
    [X,Y,T,area] = perfcurve(testLabels,Yhat,options.positiveClass);
end
end