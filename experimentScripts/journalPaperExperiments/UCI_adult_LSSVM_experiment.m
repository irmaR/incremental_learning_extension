function []=UCI_adult_LSSVM_experiment(pathToData,pathToResults,pathToCode,nrRuns,nrSamples,batchSize,dataLimit,reguGammas,kernelParams)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
addpath(genpath(pathToCode))
load(pathToData)

outputPath=sprintf('%s/smp_%d/bs_%d/%s/',pathToResults,nrSamples,batchSize,'lssvm');
fprintf('Making folder %s',outputPath)
mkdir(outputPath)

for r=1:nrRuns
    s = RandStream('mt19937ar','Seed',r);
    load(pathToData)
    %shuffle the training data with the seed according to the run
    ix=randperm(s,size(train,1))';
    %pick 60% of the data in this run to be used
    trainData=train(ix(1:ceil(size(ix,1)*2/3)),:);
    trainClass=train_class(ix(1:ceil(size(ix,1)*2/3),:));
    
    %standardize the training and test data
    [trainData,min_train,max_train]=standardizeX(trainData);
    testData=standardize(test,min_train,max_train);
    
    %this test dataset is pretty big so we will sample 1000 points in each
    %run
    ix=randperm(s,size(test,1))';
    testData=testData(ix(1:1000),:);
    testClass=test_class(ix(1:1000),:);
    
    trainClass(trainClass~=1)=-1;
    trainClass(trainClass==2)=1;
    testClass(testClass~=1)=-1;
    testClass(testClass==2)=1;
    
    fprintf('Number of training data points %d-%d, class %d\n',size(trainData,1),size(trainData,2),size(train_class,1));
    fprintf('Number of test data points %d-%d\n',size(test,1),size(test,2));
    reportPoints=[nrSamples:batchSize:size(trainData,1)-batchSize];
    
    settings.XTest=testData;
    settings.YTest=testClass;
    settings.XTrain=trainData;
    settings.YTrain=trainClass;
    settings.numSelectSamples=nrSamples;
    settings.batchSize=batchSize;
    settings.reportPoints=reportPoints;
    settings.dataLimit=dataLimit;
    settings.outputPath=outputPath;
    settings.reportPointIndex=1;
    settings.run=r;
    settings.reguGammas=reguGammas;
    settings.kernelParams=kernelParams;
    
    fprintf('Number of report points:%d',length(reportPoints))
    %we don't use validation here. We tune parameters on training data
    %(5-fold-crossvalidation)
    res=runExperiment(settings,'lssvm');%res=run_experiment(train,train_class,test,test_class,nr_samples,interval,batch_size,report_points,method,data_limit,r)
    results{r}=res;
    %save intermediate results just in case
    save(sprintf('%s/results.mat',outputPath),'results');
    avgAucs=zeros(1,length(reportPoints));
    realAvgAUCs=zeros(1,length(reportPoints));
    for i=1:r
        size(cell2mat(results{i}.selectedAUCs))
        size(reportPoints)
        avgAucs=avgAucs+cell2mat(results{i}.selectedAUCs);
        realAvgAUCs=realAvgAUCs+cell2mat(results{i}.AUCs);
        allAucs(i,:)=cell2mat(results{i}.selectedAUCs);
        allRealAucs(i,:)=cell2mat(results{i}.AUCs);
        runTimes(i,:)=results{i}.runtime+results{i}.tuningTime;
        processingTimes(i,:)=results{i}.processingTimes;
    end
    stdev=std(allAucs);
    stdevReal=std(allRealAucs);
    avgAucs=avgAucs/nrRuns;
    realAvgAUCs=realAvgAUCs/nrRuns;
    avgRuntime=mean(runTimes);
    stdRuntime=std(runTimes);
    save(sprintf('%s/auc.mat',outputPath),'avgAucs','realAvgAUCs','stdev','stdevReal','reportPoints','avgRuntime','stdRuntime','processingTimes');
end

avgAucs=zeros(1,length(reportPoints));
realAvgAUCs=zeros(1,length(reportPoints));
for i=1:nrRuns
    avgAucs=avgAucs+cell2mat(results{i}.selectedAUCs);
    realAvgAUCs=realAvgAUCs+cell2mat(results{i}.AUCs);
    allAucs(i,:)=cell2mat(results{i}.selectedAUCs);
    allRealAucs(i,:)=cell2mat(results{i}.AUCs);
    runTimes(i,:)=results{i}.runtime+results{i}.tuningTime;
    processingTimes(i,:)=results{i}.processingTimes;
end
stdev=std(allAucs);
stdevReal=std(allRealAucs);
avgAucs=avgAucs/nrRuns;
realAvgAUCs=realAvgAUCs/nrRuns;
avgRuntime=mean(runTimes);
stdRuntime=std(runTimes);
save(sprintf('%s/auc.mat',outputPath),'avgAucs','realAvgAUCs','stdev','stdevReal','reportPoints','avgRuntime','stdRuntime','processingTimes');
save(sprintf('%s/results.mat',outputPath),'results');
end

