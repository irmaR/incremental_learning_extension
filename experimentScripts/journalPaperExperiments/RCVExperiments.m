function []=RCVExperiments(method,pathToData,pathToResults,pathToCode,nrRuns,nrSamples,batchSize,dataLimit,warping,balanced,reguBetaParams,reguAlphaParams,kernelParams,k,WeightMode,NeighborMode)
%RCV mat contains train,train_class,test and test_class
%we use one vs all strategy
nargin
switch nargin
    case 13
        NeighborModes={'Supervised'};
        WeightModes={'HeatKernel','Cosine'}
        ks=[0];
    case 16
        NeighborModes={NeighborMode};
        WeightModes={WeightMode};
        ks=[k];
end
addpath(genpath(pathToCode))
load(pathToData)
reportPoints=[nrSamples:batchSize:(8664-2000)];
for ns=1:length(NeighborModes)
    for ws=1:length(WeightModes)
        for kNN=1:length(ks)
            general_output=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/',pathToResults,nrSamples,batchSize,NeighborModes{ns},WeightModes{ws},ks(kNN));
            outputPath=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/%s/',pathToResults,nrSamples,batchSize,NeighborModes{ns},WeightModes{ws},ks(kNN),method);
            fprintf('Making folder %s',outputPath)
            mkdir(outputPath)
            param_info=sprintf('%s/params.txt',outputPath)
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
            fprintf(fileID,'nr_samples:%d \n',nrSamples);
            fprintf(fileID,'batch_size:%d \n',batchSize);
            fprintf(fileID,'data_limit:%d \n',dataLimit);
            fprintf(fileID,'Using warping?:%d \n',warping);
            fprintf(fileID,'Using balancing?:%d \n',balanced);
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
                    [trainData,min_train,max_train]=standardizeX(trainData);
                    testData=standardize(testData,min_train,max_train);
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
                    settings.YTest=testClass;
                    settings.validation=validation;
                    settings.validationClass=validationClass;
                    settings.XTrain=trainData;
                    settings.YTrain=trainClass;
                    settings.reguAlphaParams=reguAlphaParams;
                    settings.reguBetaParams=reguBetaParams;
                    settings.kernelParams=kernelParams;
                    settings.numSelectSamples=nrSamples;
                    settings.batchSize=batchSize;
                    settings.reportPoints=reportPoints;
                    settings.dataLimit=dataLimit;
                    settings.run=r;
                    settings.warping=warping;
                    settings.balanced=balanced;
                    settings.weightMode=WeightModes{ws};
                    settings.neighbourMode=NeighborModes{ns};
                    settings.ks=ks(kNN);
                    settings.outputPath=outputPath;
                    settings.reportPointIndex=1;
                    settings.positiveClass=1;
                    settings.classes=[1,2];
                    res1=runExperiment(settings,method)
                    selectedAUCs(c,:)=cell2mat(res1.selectedAUCs);
                    SRKDAAucs(c,:)=cell2mat(res1.SRKDAAucs);
                    SVMAucs(c,:)=cell2mat(res1.SVMAUCs);
                    RidgeAucs(c,:)=cell2mat(res1.RidgeAUCs);
                    SRDAAucs(c,:)=cell2mat(res1.SRDAAUC);
                    DTAucs(c,:)=cell2mat(res1.DTAUCs);
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
                res.stdevRealAucs=nanstd(SRKDAAucs);
                res.stdevAucs=nanstd(selectedAUCs);
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
            size(avgAucs)
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
            stdevSVMAucs=nanstd(SVMAucs);
            DTAucs=nanmean(DTAucs);
            stdevDTAucs=nanstd(DTAucs);
            RidgeAucs=nanmean(RidgeAucs);
            stdevRidgeAucs=nanstd(RidgeAucs);
            save(sprintf('%s/auc.mat',outputPath),'SRDAAucs','stdevSRDAAucs','SVMAucs','stdevSVMAucs','DTAucs','stdevDTAucs','RidgeAucs','stdevRidgeAucs','avgAucs','realAvgAUCs','stdev','stdevReal','reportPoints','avgRuntime','stdRuntime','processingTimes');
            save(sprintf('%s/results.mat',outputPath),'results');
        end
    end
end
