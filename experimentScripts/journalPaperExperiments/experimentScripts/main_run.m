function [res]=main_run(experimentName,fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,numSelectSamples,batchSize,dataLimit,validationOffset,runs)
% fidTrain='/home/irma/work/DATA/HEPMASS/1000000_reduced/train_200.csv';
% fidTest='/home/irma/work/DATA/HEPMASS/1000000_reduced/1000_test_reduced.csv';
% offsetTrain='/home/irma/work/DATA/HEPMASS/1000000_reduced/train_200_offset.mat';
% offsetTest='/home/irma/work/DATA/HEPMASS/1000000_reduced/reduced_test_offsetIndices.mat';
% output='/home/irma/work/RESULTS/HEPMASS/1000000_reduced/smp_50/';
% codePath='/home/irma/work/CODE/incremental_learning_extension/';
w = warning ('off','all');
resultsPath=highMass(experimentName,fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,runs,validationOffset,numSelectSamples,batchSize,dataLimit)
end