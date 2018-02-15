fidTrain='/home/irma/work/DATA/HEPMASS/1000000_reduced/1000000_train.csv';
fidTest='/home/irma/work/DATA/HEPMASS/1000000_reduced/1000_test_reduced.csv';
offsetTrain='/home/irma/work/DATA/HEPMASS/1000000_reduced/1000000_train_offset.mat';
offsetTest='/home/irma/work/DATA/HEPMASS/1000000_reduced/reduced_test_offsetIndices.mat';
output='/home/irma/work/RESULTS/HEPMASS/1000000_reduced/smp_50/';
codePath='/home/irma/work/CODE/incremental_learning_extension/';
numSelectSamples=50
batchSize=200;
dataLimit=3000;
validationOffset=3000;
runs=1;

resultsPath=highMassRandom(fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,runs,validationOffset,numSelectSamples,batchSize,dataLimit)