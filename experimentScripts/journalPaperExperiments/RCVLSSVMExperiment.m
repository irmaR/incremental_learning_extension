function []=RCVLSSVMExperiment(pathToData,pathToResults,pathToCode,nrRuns,nrSamples,batchSize,dataLimit,reguGammas,kernelParams)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
addpath(genpath(pathToCode))
load(pathToData)


outputPath=sprintf('%s/smp_%d/bs_%d/%s/',pathToResults,nrSamples,batchSize,'lssvm');
fprintf('Making folder %s',outputPath)
mkdir(outputPath)
reportPoints=[nrSamples:batchSize:(8664-2000)];
for r=1:nrRuns
    aucs=[];
    tuning_time=[];
    runtime=[];
    res=[];
    for c=1:4
        trainData=folds{r}.train;
        trainClass=folds{r}.train_class;
        testData=folds{r}.test;
        testClass=folds{r}.test_class;
        
        %shuffle data according to the seed of run
        s = RandStream('mt19937ar','Seed',r);
        
        %shuffle the training data with the seed according to the run
        ix=randperm(s,size(trainData,1))';
        trainData=trainData(ix,:);
        trainClass=trainClass(ix,:);
        
        %standardize the training and test data
        trainData=standardizeX(trainData);
        testData=standardizeX(testData);
        %for each category in train class we run one learning/inference
        %procedure. We calculate AUCs and we average then
        fprintf('Number of training data points %d-%d, class %d\n',size(trainData,1),size(trainData,2),size(trainClass,1));
        fprintf('Number of test data points %d-%d\n',size(testData,1),size(testData,2));
        
        
        for v=1:size(trainClass,1)
            if trainClass(v,:)==c
                trainClass(v,:)=1;
            else
                trainClass(v,:)=2;
            end
        end
        
        for v=1:size(testClass,1)
            if testClass(v,:)==c
                testClass(v,:)=1;
            else
                testClass(v,:)=2;
            end
        end
        
        %I need validation data (a random subset of train data for
        %model selection)
        validation=trainData(1:2000,:);
        validationClass=trainClass(1:2000,:);
        trainData=trainData(2001:end,:);
        trainClass=trainClass(2001:end,:);
        
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
        res1=runExperiment(settings,'lssvm')
        
        selectedAUCs(c,:)=cell2mat(res1.selectedAUCs);
        SRKDAAucs(c,:)=cell2mat(res1.SRKDAAucs);
        SVMAucs(c,:)=cell2mat(res1.SVMAUCs);
        RidgeAucs(c,:)=cell2mat(res1.RidgeAUCs);
        SRDAAucs(c,:)=cell2mat(res1.SRDAAUC);
        DTAucs(c,:)=cell2mat(res1.DTAUCs);
        realAUCs(c,:)=cell2mat(res1.AUCs);
        tuningTime(c,:)=res1.tuningTime;
        runtime(c,:)=res1.runtime;
        processingTime(c,:)=res1.processingTimes;
    end
    res.SVMAucs=nanmean(SVMAucs);
    res.SRDAAucs=nanmean(SRDAAucs);
    res.DTAucs=nanmean(DTAucs);
    res.RidgeAucs=nanmean(RidgeAucs);
    res.stdevSVMAuc=nanstd(SVMAucs);
    res.stdevDTAucs=nanstd(DTAucs);
    res.stdevRidgeAucs=nanstd(RidgeAucs);
    res.stdevSRDAAucs=nanstd(SRDAAucs);
    res.avgAUCs=nanmean(selectedAUCs);
    res.SRKDAAucs=nanmean(SRKDAAucs);
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
    avgAucs(i,:)=results{i}.avgAUCs;
    SRKDAAucs(i,:)=results{i}.SRKDAAucs;
    SVMAucs(i,:)=results{i}.SVMAucs;
    DTAucs(i,:)=results{i}.DTAucs;
    SRDAAucs(i,:)=results{i}.SRDAAucs;
    RidgeAucs(i,:)=results{i}.RidgeAucs;
    allAucs(i,:)=results{i}.avgAUCs;
    allRealAucs(i,:)=results{i}.SRKDAAucs;
    runTimes(i,:)=results{i}.runtime+results{i}.tuningTime;
    processingTimes(i,:)=results{i}.processingTime;
end
stdev=nanstd(allAucs);
stdevReal=nanstd(SRKDAAucs);
avgAucs=nanmean(avgAucs,1)
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
save(sprintf('%s/auc.mat',outputPath),'SRKDAAucs','stdevSRKDAAucs','SRDAAucs','stdevSRDAAucs','SVMAucs','stdevSVMAucs','DTAucs','stdevDTAucs','RidgeAucs','stdevRidgeAucs','avgAucs','realAvgAUCs','stdev','stdevReal','reportPoints','avgRuntime','stdRuntime','processingTimes');
save(sprintf('%s/results.mat',outputPath),'results');
end
