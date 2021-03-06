function []=USPS_LSSVM(pathToData,pathToResults,pathToCode,nrRuns,nrSamples,batchSize,dataLimit,reguGammas,kernelParams)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
addpath(genpath(pathToCode))
load(pathToData)


outputPath=sprintf('%s/smp_%d/bs_%d/%s/',pathToResults,nrSamples,batchSize,'lssvm');
fprintf('Making folder %s',outputPath)
mkdir(outputPath)
reportPoints=[nrSamples:batchSize:8400];
for r=1:nrRuns
    aucs=[];
    tuningTime=[];
    runtime=[];
    res=[];
    for c=1:10
        trainData=folds{r}.train;
        trainClass=folds{r}.train_class;
        testData=folds{r}.test;
        testClass=folds{r}.test_class;
        %standardize the training and test data
        trainData=standardizeX(trainData);
        testData=standardizeX(testData);
        %for each category in train class we run one learning/inference
        %procedure. We calculate AUCs and we average then
        fprintf('Number of training data points %d-%d, class %d\n',size(trainData,1),size(trainData,2),size(trainClass,1));
        trainClass(trainClass~=c)=-1;
        trainClass(trainClass==c)=1;
        testClass(testClass~=c)=-1;
        testClass(testClass==c)=1;
        
        settings.XTest=testData;
        settings.YTest=testClass;
        settings.XTrain=trainData;
        settings.YTrain=trainClass;
        fprintf('Number of test data points %d-%d, class %d\n',size(settings.XTest,1),size(settings.XTest,2),size(settings.YTest,2));
        settings.reguGammas=reguGammas;
        settings.kernelParams=kernelParams;
        settings.numSelectSamples=nrSamples;
        settings.batchSize=batchSize;
        settings.reportPoints=reportPoints;
        settings.dataLimit=dataLimit;
        settings.run=r;
        settings.outputPath=outputPath;
        settings.reportPointIndex=1;
        res1=runExperiment(settings,'lssvm')
        selectedAUCs(c,:)=cell2mat(res1.selectedAUCs);
        realAUCs(c,:)=cell2mat(res1.AUCs);
        tuningTime(c,:)=res1.tuningTime;
        runtime(c,:)=res1.runtime;
        processingTime(c,:)=res1.processingTimes;
    end
    res.avgAUCs=mean(selectedAUCs);
    res.avgRealAUCs=mean(realAUCs);
    res.stdevRealAucs=std(realAUCs);
    res.stdevAucs=std(selectedAUCs);
    res.reportPoints=reportPoints;
    res.tuningTime=mean(tuningTime);
    res.stdevTuningTime=std(tuningTime);
    res.runtime=mean(runtime);
    res.stdevRuntime=std(runtime);
    res.processingTime=mean(processingTime);
    res.avgRuntime=mean(runtime);
    res.stdRuntime=std(runtime);
    save(sprintf('%s/res.mat',outputPath),'res');
    results{r}=res;
end
avgAucs=zeros(1,length(reportPoints));
realAvgAUCs=zeros(1,length(reportPoints));
for i=1:nrRuns
    avgAucs=avgAucs+results{i}.avgAUCs;
    realAvgAUCs=realAvgAUCs+results{i}.avgRealAUCs;
    allAucs(i,:)=results{i}.avgAUCs;
    allRealAucs(i,:)=results{i}.avgRealAUCs;
    runTimes(i,:)=results{i}.runtime+results{i}.tuningTime;
    processingTimes(i,:)=results{i}.processingTime;
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
