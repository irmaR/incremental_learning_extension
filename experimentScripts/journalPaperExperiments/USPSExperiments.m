function []=USPSExperiments(method,pathToData,pathToResults,pathToCode,nrRuns,nrSamples,batchSize,dataLimit,warping,balanced,reguBetaParams,reguAlphaParams,kernelParams,k,WeightMode,NeighborMode)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
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
            fprintf(fileID,'\n')
            fprintf(fileID,'Nr runs:%d \n',nrRuns);
            fprintf(fileID,'nr_samples:%d \n',nrSamples);
            fprintf(fileID,'batch_size:%d \n',batchSize);
            fprintf(fileID,'data_limit:%d \n',dataLimit);
            fprintf(fileID,'Using warping?:%d \n',warping);
            fprintf(fileID,'Using balancing?:%d \n',balanced);
            reportPoints=[nrSamples:batchSize:8400];
            for r=1:nrRuns
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
                    res1=runExperiment(settings,method)       
                    selectedAUCs(c,:)=cell2mat(res1.selectedAUCs);
                    realAUCs(c,:)=cell2mat(res1.AUCs);
                    tuningTime(c,:)=res1.tuningTime;
                    runtime(c,:)=res1.runtime;
                    processingTime(c,:)=res1.processingTimes;
                end
                res.avgAUCs=nanmean(selectedAUCs);
                res.avgRealAUCs=nanmean(realAUCs);
                res.stdevRealAucs=nanstd(realAUCs);
                res.stdevAucs=nanstd(selectedAUCs);
                res.reportPoints=reportPoints;
                res.tuningTime=nanmean(tuningTime);
                res.stdevTuningTime=nanstd(tuningTime);
                res.runtime=nanmean(runtime);
                res.stdevRuntime=nanstd(runtime);
                res.processingTime=nanmean(processingTime);
                res.avgRuntime=nanmean(runtime);
                res.stdRuntime=nanstd(runtime);
                save(sprintf('%s/res.mat',outputPath),'res');
                results{r}=res;
            end
            avgAucs=zeros(1,length(reportPoints));
            realAvgAUCs=zeros(1,length(reportPoints));
            for i=1:nrRuns
                results{i}.avgAUCs
                avgAucs=avgAucs+results{i}.avgAUCs;               
                realAvgAUCs=realAvgAUCs+results{i}.avgRealAUCs;
                allAucs(i,:)=results{i}.avgAUCs;
                allRealAucs(i,:)=results{i}.avgRealAUCs;
                runTimes(i,:)=results{i}.runtime+results{i}.tuningTime;
                processingTimes(i,:)=results{i}.processingTime;
            end
            stdev=nanstd(allAucs);
            stdevReal=nanstd(allRealAucs);
            avgAucs=avgAucs/nrRuns;
            realAvgAUCs=realAvgAUCs/nrRuns;
            avgRuntime=nanmean(runTimes);
            stdRuntime=nanstd(runTimes);
            save(sprintf('%s/auc.mat',outputPath),'avgAucs','realAvgAUCs','stdev','stdevReal','reportPoints','avgRuntime','stdRuntime','processingTimes');
            save(sprintf('%s/results.mat',outputPath),'results');
        end
    end
end
