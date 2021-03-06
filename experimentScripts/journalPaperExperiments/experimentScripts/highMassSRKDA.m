function [outputPath]=highMassSRKDA(fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,runs,validationOffset,numSelectSamples,batchSize,dataLimit)
%fidTrain='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/train/all_train.csv';
%fidTest='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/test/all_test.csv';
%offsetTrain='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/offsetIndices.mat';
%offsetTest='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/offsetIndicesTest.mat';
%output='/Users/irma/Documents/MATLAB/RESULTS/Test/HighMassNoMass/samples10000/';
%codePath='/Users/irma/Documents/MATLAB/CODE_local/incremental_learning_extension';

shuffleSeedValidation=1000;
outputPath=sprintf('%s/smp_%d/bs_%d/',output,numSelectSamples,batchSize);
validationPath=sprintf('%s/smp_%d/bs_%d/iSRKDA/validation/parameters.mat',output,numSelectSamples,batchSize)
if exist(validationPath, 'file') == 2
    params=load(validationPath);
    params=params.parameters;
else
    fprintf('Validation parameters of iSRKDA not determined. run validation...');
    params=highMassPhysicsExperimentValidation('iSRKDA',@MAEDIncrementalSequential,shuffleSeedValidation,validationOffset,@srkdaInference,fidTrain,offsetTrain,outputPath,codePath,numSelectSamples,batchSize,dataLimit,1,1,[0.0001,0.01],[0.001],[5,10]);
end
for run=1:runs
    highMassPhysicsExperimenttraining('SRKDA',run,shuffleSeedValidation,validationOffset,fidTrain,fidTest,offsetTrain,offsetTest,outputPath,codePath,numSelectSamples,batchSize,dataLimit,1,1,[params.reguBeta],[params.reguAlpha],[params.kernelSigma]);
end
