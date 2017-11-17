function [outputPath]=highMassiSRKDA(fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,runs,validationOffset,numSelectSamples,batchSize,dataLimit)
shuffleSeedValidation=1000;
outputPath=sprintf('%s/smp_%d/bs_%d/',output,numSelectSamples,batchSize)
params=highMassPhysicsExperimentValidation('iSRKDA',@MAEDIncrementalSequential,shuffleSeedValidation,validationOffset,@srkdaInference,fidTrain,offsetTrain,outputPath,codePath,numSelectSamples,batchSize,dataLimit,1,1,[0.0001,0.01],[0.001],[5,10]);
for run=1:runs
    highMassPhysicsExperimenttraining('iSRKDA',run,shuffleSeedValidation,validationOffset,fidTrain,fidTest,offsetTrain,offsetTest,outputPath,codePath,numSelectSamples,batchSize,dataLimit,1,1,[params.reguBeta],[params.reguAlpha],[params.kernelSigma]);
    %highMassPhysicsExperimenttraining('SRKDA',run,shuffleSeedValidation,validationOffset,fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,50,200,2000,1,1,[params.reguBeta],[params.reguAlpha],[params.kernelSigma]);
end


