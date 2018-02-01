fidTrain='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/train/all_train.csv';
fidTest='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/test/all_test.csv';
offsetTrain='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/offsetIndices.mat';
offsetTest='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/offsetIndicesTest.mat';
output='/Users/irma/Documents/MATLAB/RESULTS/Test/HighMass/';
codePath='/Users/irma/Documents/MATLAB/CODE_local/incremental_learning_extension';
numSelectSamples=50;
batchSize=200;
dataLimit=3000;
validationOffset=3000;
runs=2;

%resultsPath=highMassiSRKDA(fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,runs,validationOffset,numSelectSamples,batchSize,dataLimit);
%highMassPlotAUCs(resultsPath,runs)

%resultsPath=highMassRandom(fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,runs,validationOffset,numSelectSamples,batchSize,dataLimit);
%highMassPlotAUCs(resultsPath,runs)

%resultsPath=highMassSRKDA(fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,runs,validationOffset,numSelectSamples,batchSize,dataLimit);


res='/Users/irma/Documents/MATLAB/RESULTS/ResultsNovember/Incremental/RCV/smp_10/bs_100/';
methods={'SRKDA','iSRKDA','lssvm','random'};
colors('iSRKDA')={'blue'};
colors('SRKDA')={'red'};
colors('lssvm')={'black'};
colors('random')={'green'};
%plotAUCvsObservedPoints(res,methods,colors,runs)
plotAUCvsObservedPoints(res,methods)
%resultsPath=highMassRandom(fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,runs,validationOffset,numSelectSamples,batchSize,dataLimit)