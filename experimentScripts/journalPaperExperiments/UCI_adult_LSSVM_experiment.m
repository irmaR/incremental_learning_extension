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
    %I need validation data (a random subset of train data for
    %model selection)
    validation=trainData(1:2000,:);
    validationClass=trainClass(1:2000,:);
    trainData=trainData(2001:end,:);
    trainClass=trainClass(2001:end,:);
    
    %this test dataset is pretty big so we will sample 1000 points in each
    %run
    ix=randperm(s,size(test,1))';
    testData=testData(ix(1:1000),:);
    testClass=test_class(ix(1:1000),:);
    
    
    fprintf('Number of training data points %d-%d, class %d\n',size(trainData,1),size(trainData,2),size(trainClass,1));
    fprintf('Number of test data points %d-%d\n',size(testData,1),size(testData,2));
    reportPoints=[nrSamples:batchSize:size(trainData,1)-batchSize];
    
    settings.XTest=testData;
    settings.markSelPoints=1;
    settings.YTest=testClass;
    settings.validation=validation;
    settings.validationClass=validationClass;
    settings.XTrain=trainData;
    settings.YTrain=trainClass;
    settings.kernelParams=kernelParams;
    settings.numSelectSamples=nrSamples;
    settings.batchSize=batchSize;
    settings.reportPoints=reportPoints;
    settings.dataLimit=dataLimit;
    settings.run=r;
    settings.reguGammas=reguGammas;
    settings.kernelType='RBF_kernel';
    settings.gamma=1;
    settings.outputPath=outputPath;
    settings.reportPointIndex=1;
    settings.positiveClass=1;
    settings.classes=[1,2];
    
    fprintf('Number of report points:%d',length(reportPoints))
    %we don't use validation here. We tune parameters on training data
    %(5-fold-crossvalidation)
    res=runExperiment(settings,'lssvm');%res=run_experiment(train,train_class,test,test_class,nr_samples,interval,batch_size,report_points,method,data_limit,r)
    results{r}=res;
    %save intermediate results just in case
    save(sprintf('%s/results.mat',outputPath),'results');
end
for i=1:nrRuns
    SRKDAAucs(i,:)=cell2mat(results{i}.SRKDAAucs);
    SVMAucs(i,:)=cell2mat(results{i}.SVMAUCs);
    DTAucs(i,:)=cell2mat(results{i}.DTAUCs);
    SRDAAucs(i,:)=cell2mat(results{i}.SRDAAUC);
    RidgeAucs(i,:)=cell2mat(results{i}.RidgeAUCs);
    runTimes(i,:)=results{i}.runtime+results{i}.tuningTime;
    processingTimes(i,:)=results{i}.processingTimes;
end
stdevReal=nanstd(SRKDAAucs);
SRKDAAucs=nanmean(SRKDAAucs,1);
avgRuntime=mean(runTimes);
stdRuntime=std(runTimes);
SVMAucs=nanmean(SVMAucs);
SRDAAucs=nanmean(SRDAAucs);
stdevSRDAAucs=nanstd(SRDAAucs);
stdevSRKDAAucs=nanstd(SRKDAAucs);
stdevSVMAucs=nanstd(SVMAucs);
DTAucs=nanmean(DTAucs);
stdevDTAucs=nanstd(DTAucs);
RidgeAucs=nanmean(RidgeAucs);
stdevRidgeAucs=nanstd(RidgeAucs);
save(sprintf('%s/auc.mat',outputPath),'SRKDAAucs','stdevSRKDAAucs','SRDAAucs','stdevSRDAAucs','SVMAucs','stdevSVMAucs','DTAucs','stdevDTAucs','RidgeAucs','stdevRidgeAucs','reportPoints','avgRuntime','stdRuntime','processingTimes');
save(sprintf('%s/results.mat',outputPath),'results');
end
