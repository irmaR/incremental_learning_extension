function [outputPath]=highMassRandom(fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,runs,validationOffset,numSelectSamples,batchSize,dataLimit)
addpath(genpath(codePath))
shuffleSeedValidation=1000;
outputPath=sprintf('%s/smp_%d/bs_%d/',output,numSelectSamples,batchSize);
validationPath=sprintf('%s/smp_%d/bs_%d/random/validation/parameters.mat',output,numSelectSamples,batchSize)
%if exist(validationPath, 'file') == 2
%    params=load(validationPath);
%    params=params.parameters;
%else
%    fprintf('Validation parameters of iSRKDA not determined. run validation...');
%    params=highMassPhysicsExperimentValidation('random',@MAEDIncrementalSequential,shuffleSeedValidation,validationOffset,@srkdaInference,fidTrain,offsetTrain,outputPath,codePath,numSelectSamples,batchSize,dataLimit,1,1,[0.0001,0.01],[0.001],[5,10]);
%end
params.reguBeta=0.01
params.reguAlpha=0.01
params.kernelSigma=0.5
for run=1:runs
    highMassPhysicsExperimenttraining('random',run,shuffleSeedValidation,validationOffset,fidTrain,fidTest,offsetTrain,offsetTest,outputPath,codePath,numSelectSamples,batchSize,dataLimit,1,1,[params.reguBeta],[params.reguAlpha],[params.kernelSigma]);
end
