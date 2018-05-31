function [outputPath]=highMass(experimentName,fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,runs,validationOffset,numSelectSamples,batchSize,dataLimit)
shuffleSeedValidation=1000;
outputPath=sprintf('%s/smp_%d/bs_%d/',output,numSelectSamples,batchSize)
%params=highMassPhysicsExperimentValidation('iSRKDA',@MAEDIncrementalSequential,shuffleSeedValidation,validationOffset,@srkdaInference,fidTrain,offsetTrain,outputPath,codePath,numSelectSamples,batchSize,dataLimit,1,1,[0.0001,0.01],[0.001],[5,10]);
fprintf('Validation finished')
addpath(genpath(codePath));
params.reguAlpha=[0.1,0.01];
params.reguBeta=[0.1,0.01];
params.kernelSigma=[0.5,1];
for run=1:runs
    highMassPhysicsExperimenttraining(experimentName,run,shuffleSeedValidation,validationOffset,fidTrain,fidTest,offsetTrain,offsetTest,outputPath,numSelectSamples,batchSize,dataLimit,1,1,[params.reguBeta],[params.reguAlpha],[params.kernelSigma]);
end


