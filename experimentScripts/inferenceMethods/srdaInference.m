function [area]=srdaInference(K,trainFea,trainLabels,testFea,testLabels,options)
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
[eigvector, ~] = SR_caller(options,trainFea);
%ClassLabel = unique(trainLabels);
%nClass = length(ClassLabel);
%TrainEmbed=trainFea*eigvector;
%ClassCenter = zeros(nClass,size(TrainEmbed,2));
%for i = 1:nClass
%    feaTmp = TrainEmbed(trainLabels == ClassLabel(i),:);
%    ClassCenter(i,:) = mean(feaTmp,1);
%end
Yhat = testFea*eigvector; %projection
%D = EuDist2(Yhat,ClassCenter);
%[dump, idx] = min(D,[],2);
%predictlabel = ClassLabel(idx);
%accuracy = 1 - length(find(predictlabel-testLabels))/size(testFea,1);
if sum(isnan(Yhat))~=0
    area=0;
else
    [X,Y,T,area] = perfcurve(testLabels,Yhat,'1');
end
end