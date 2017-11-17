function []=spiralsRuntimeExperiment(method,pathToData,pathtoTest,pathToResults,pathToCode,numSelectSamples,batchSize,dataLimit,warping,balanced,betas,alphas,kernels)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
addpath(genpath(pathToCode))
reguBetaParams=betas;
reguAlphaParams=alphas;
kernelParams=kernels;
NeighborModes='Supervised';
WeightModes='HeatKernel'
ks=0;

for r=1:size(numSelectSamples,2)
    output_path=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/%s/',pathToResults,numSelectSamples(r),batchSize,method);
    fprintf('Making folder %s',output_path)
    mkdir(output_path)
    param_info=sprintf('%s/params.txt',output_path)
    fileID = fopen(param_info,'w');
    fprintf(fileID,'Beta params=: ');
    for i=1:length(reguBetaParams)
        fprintf(fileID,'%1.3f',reguBetaParams(i));
    end
    fprintf(fileID,'\n');
    fprintf(fileID,'Alpha params: ');
    for i=1:length(reguAlphaParams)
        fprintf(fileID,'%1.3f',reguAlphaParams(i));
    end
    fprintf(fileID,'\n');
    fprintf(fileID,'Kernel params: ');
    for i=1:length(kernelParams)
        fprintf(fileID,'%1.3f',kernelParams(i));
    end
    fprintf(fileID,'nr_samples:%d \n',numSelectSamples(r));
    fprintf(fileID,'batch_size:%d \n',batchSize);
    fprintf(fileID,'data_limit:%d \n',dataLimit);
    fprintf(fileID,'Using warping?:%d \n',warping);
    fprintf(fileID,'Using balancing?:%d \n',balanced);
    
    trainData=load(pathToData);
    testData=load(pathtoTest);
    %shuffle the training data with the seed according to the run
    ix=randperm(size(trainData.fea,1))';
    %pick 60% of the data in this run to be used
    train=trainData.fea(ix,:);
    trainClass=trainData.gnd(ix,:);
    test=testData.fea;
    testClass=testData.gnd; %we only check the runtime
    %standardize the training and test data
    train=standardizeX(train);
    test=standardizeX(test);

    fprintf('Number of training data points %d-%d, class %d\n',size(train,1),size(train,2),size(trainClass,1));
    fprintf('Number of test data points %d-%d\n',size(test,1),size(test,2));
    reportPoints=[numSelectSamples(r):batchSize:size(train,1)-batchSize]
    fprintf('Number of report points:%d',length(reportPoints))
    %we don't use validation here. We tune parameters on training data
    %(5-fold-crossvalidation)
    settings.XTrain=train;
    settings.YTrain=trainClass;
    settings.XTest=test;
    settings.YTest=testClass;
    settings.initSample=[];
    settings.initClass=[];
    settings.reportPointIndex=1;
    settings.reguAlphaParams=reguAlphaParams;
    settings.reguBetaParams=reguBetaParams;
    settings.kernelParams=kernelParams;
    settings.numSelectSamples=numSelectSamples(r);
    settings.batchSize=batchSize;
    settings.reportPoints=reportPoints;
    settings.dataLimit=dataLimit;
    settings.run=r;
    settings.warping=warping;
    settings.balanced=balanced;
    settings.weightMode=WeightModes;
    settings.neighbourMode=NeighborModes;
    settings.ks=ks;    
    res=runExperiment(settings,method);
    runtime=res.runtime;
    %save intermediate results just in case
    save(sprintf('%s/results.mat',output_path),'res');
    save(sprintf('%s/auc.mat',output_path),'reportPoints','runtime');
end
end