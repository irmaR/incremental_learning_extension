function []=two_circles_experiment()
%clear all;
%close all;
%clc;
addpath(genpath('/Users/irma/Documents/MATLAB/incremental_learning'))
reguBetaParams=[0.01,0.04];
reguAlphaParams=[0.01,0.04];
kernelParams=[0.5,5];
reguGammas=[1];

interval=5;
nrSamples=20;
batchSize=40;
dataLimit=400;
warping=1;
blda=1;
NeighborModes={'Supervised'};
WeightModes={'HeatKernel','Cosine'}
ks=[0];
nrRuns=1;

for r=1:nrRuns
    s = RandStream('mt19937ar','Seed',r);
    rand('twister',r*1000);
    [train, trainClass] = GenTwoNoisyCircle(500,70);
    rand('twister',r*2*1000);
    [test, testClass] = GenTwoNoisyCircle(50,60);
    rand('twister',r*3*1000);
    reportPoints=[nrSamples:batchSize:size(train,1)-batchSize-interval]
    %shuffle the training data
    ix=randperm(s,size(train,1))';
    train=train(ix,:);
    trainClass=trainClass(ix,:);
    %train=NormalizeFea(train);
    %test=NormalizeFea(test);
    train=standardizeX(train);
    test=standardizeX(test);
    %test=train;
    %test_class=train_class;
    rand('twister',r*4*1000);
    [validation, validationClass] = GenTwoNoisyCircle(60,60);
    validation=standardizeX(validation);
    
    settings.XTest=test;
    settings.markSelPoints=1;
    settings.YTest=testClass;
    settings.validation=validation;
    settings.validationClass=validationClass;
    settings.XTrain=train;
    settings.YTrain=trainClass;
    settings.reguAlphaParams=reguAlphaParams;
    settings.reguBetaParams=reguBetaParams;
    settings.reguGammas=reguGammas;
    settings.kernelParams=kernelParams;
    settings.kernelType='RBF_kernel';
    settings.numSelectSamples=nrSamples;
    settings.batchSize=batchSize;
    settings.reportPoints=reportPoints;
    settings.dataLimit=dataLimit;
    settings.run=1;
    settings.warping=warping;
    settings.bLDA=blda;
    settings.weightMode=WeightModes{1};
    settings.neighbourMode=NeighborModes{1};
    settings.ks=ks(1);
    settings.gamma=1;
    %settings.outputPath=outputPath;
    settings.reportPointIndex=1;
    settings.positiveClass=1;
    settings.classes=[1,2];
    resoMAED=runExperiment(settings,'iSRKDA');
    resbMAED=runExperiment(settings,'SRKDA');
    resFLSSVM=runExperiment(settings,'lssvm');
    resSRMS=runExperiment(settings,'srms');
    resRandom=runExperiment(settings,'random');
    
    
    resultsoMAED{r}=resoMAED;
    resultsbMAED{r}=resbMAED;
    resultsFLSSVM{r}=resFLSSVM;
    resultsSRMS{r}=resSRMS;
    resultsRandom{r}=resRandom;
end
avgAucsoMAED=zeros(1,length(reportPoints));
avgAucsbMAED=zeros(1,length(reportPoints));
avgAucsoFLSSVM=zeros(1,length(reportPoints));
avgAucsoSRMS=zeros(1,length(reportPoints));
avgAucsoRandom=zeros(1,length(reportPoints));

for i=1:nrRuns
    avgAucsoMAED(i,:)=resultsoMAED{i}.avgAUCs;
    avgAucsbMAED(i,:)=resultsbMAED{i}.aucs;
    avgAucsoFLSSVM(i,:)=resultsFLSSVM{i}.aucs;
    avgAucsoSRMS(i,:)=resultsSRMS{i}.aucs;
    avgAucsoRandom(i,:)=resultsRandom{i}.aucs;
end
avgAucsoMAED=nanmean(avgAucsoMAED,1);
stdAucsoMAED=nanstd(avgAucsoMAED,1);

avgAucsbMAED=nanmean(avgAucsbMAED,1);
stdAucsbMAED=nanstd(avgAucsbMAED,1);

avgAucsoFLSSVM=nanmean(avgAucsoFLSSVM,1);
stdAucsoFLSSVM=nanstd(avgAucsoFLSSVM,1);

avgAucsoSRMS=nanmean(avgAucsoSRMS,1);
stdAucsoSRMS=nanstd(avgAucsoSRMS,1);

avgAucsoRandom=nanmean(avgAucsoRandom,1);
stdAucsoRandom=nanstd(avgAucsoRandom,1);


outputPath=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/%s/',pathToResults,nrSamples,batchSize,NeighborModes{1},WeightModes{1},ks(1),'iSRKDA');
fprintf('Making folder %s',outputPath)
mkdir(outputPath)
save(sprintf('%s/auc.mat',outputPath),'avgAucsoMAED','stdAucsoMAED','report_points');
save(sprintf('%s/results.mat',outputPath),'resultsoMAED');

outputPath=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/%s/',pathToResults,nrSamples,batchSize,NeighborModes{1},WeightModes{1},ks(1),'SRKDA');
fprintf('Making folder %s',outputPath)
mkdir(outputPath)
save(sprintf('%s/auc.mat',outputPath),'avgAucsbMAED','avgAucsbMAED','report_points');
save(sprintf('%s/results.mat',outputPath),'resultsbMAED');

outputPath=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/%s/',pathToResults,nrSamples,batchSize,NeighborModes{1},WeightModes{1},ks(1),'lssvm');
fprintf('Making folder %s',outputPath)
mkdir(outputPath)
save(sprintf('%s/auc.mat',outputPath),'avgAucsoFLSSVM','stdAucsoFLSSVM','report_points');
save(sprintf('%s/results.mat',outputPath),'resultsFLSSVM');

outputPath=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/%s/',pathToResults,nrSamples,batchSize,NeighborModes{ns},WeightModes{ws},ks(kNN),'srms');
fprintf('Making folder %s',outputPath)
mkdir(outputPath)
save(sprintf('%s/auc.mat',outputPath),'avgAucsoSRMS','stdAucsoSRMS','report_points');
save(sprintf('%s/results.mat',outputPath),'resultsSRMS');

outputPath=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/%s/',pathToResults,nrSamples,batchSize,NeighborModes{ns},WeightModes{ws},ks(kNN),'random');
fprintf('Making folder %s',outputPath)
mkdir(outputPath)
save(sprintf('%s/auc.mat',outputPath),'avgAucsoRandom','stdAucsoRandom','report_points');
save(sprintf('%s/results.mat',outputPath),'resultsRandom');

