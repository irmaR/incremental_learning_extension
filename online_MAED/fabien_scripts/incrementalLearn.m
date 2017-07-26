function [model] = incrementalLearn(data, labels, options)
% Incremental Learning
%
% Input:
%   data               - Data matrix NXM, where N is the number of data
%                          points and M is then number of variables
%   labels             - Label/response for each data point (Nx1)
%   options            - Struct value: 
%       modelSize (default: 25)  - number of samples in the model
%       batchSize (default: 100) - number of samples in each training batch
%       nbTrainingSamples  (default: N)  - total bunber of training samples 
%       W       -  Affinity matrix. You can either call
%                                 "constructW" to construct the W, or
%                                 construct it by yourself.
%                                 If 'W' is not provided and 'ReguBeta'>0 ,
%                                 MAED will build a k-NN graph with Heat kernel
%                                 weight, where 'k' is a prameter.
%
%       k       -  The parameter for k-NN graph (Default is 5)
%                                 If 'W' is provided, this parameter will be
%                                 ignored.
%
%       ReguBeta    -  regularization parameter for manifold
%                                 adaptive kernel.
%
%       ReguAlpha   -  ridge regularization paramter. Default 0.01
%
%       supervised  - if set, the construction of W will take the data labels
%                     into account
%       
%
% Output:
%   model              - selected data points 
%       model.X        - input features
%       model.Y        - labels/responses
%

if(isfield(options,'batchSize'))
    batchSize = options.batchSize;
else
    batchSize = 100;
end

if(isfield(options,'modelSize'))
    modelSize = options.modelSize;
else
    modelSize = 25;
end

if(isfield(options,'nbTrainingSamples'))
    nbTrainingSamples = options.nbTrainingSamples;
else
    nbTrainingSamples = size(data,1);
end

model.X = [];
model.Y = [];

for j=1:batchSize:nbTrainingSamples-batchSize
    candidates.X = [model.X; data(j:j+batchSize-1,:)];
    candidates.Y = [model.Y; labels(j:j+batchSize-1,:)];
    
    model = MAEDRanking(candidates, modelSize, model,options);    
    
    %model.X = candidates.X(rank,:);
    %model.Y = candidates.Y(rank,:);
end


