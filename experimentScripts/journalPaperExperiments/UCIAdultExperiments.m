function []=UCIAdultExperiments(method,pathToData,pathToResults,pathToCode,nrRuns,nrSamples,batchSize,dataLimit,warping,balanced,reguBetaParams,reguAlphaParams,kernelParams,k,WeightMode,NeighborMode)
%UCI Adult mat contains train,train_class,test and test_class
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
            fprintf('%d, %s, %s\n',ks(kNN),WeightModes{ws},NeighborModes{ns})
            if ks(kNN)==0 && strcmp(WeightModes{ws},'Binary') && strcmp(NeighborModes{ns},'kNN')
                continue
            end
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
            
            
            for r=1:nrRuns
                s = RandStream('mt19937ar','Seed',r);
                load(pathToData)
                %shuffle the training data with the seed according to the run
                ix=randperm(s,size(train,1))';
                %pick 60% of the data in this run to be used
                train=train(ix(1:ceil(size(ix,1)*2/3)),:);
                trainClass=train_class(ix(1:ceil(size(ix,1)*2/3),:));
                %standardize the training and test data
                [train,min_train,max_train]=standardizeX(train);
                testData=standardize(test,min_train,max_train);
                %this test dataset is pretty big so we will sample 1000 points in each
                %run
                ix=randperm(s,size(testData,1))';
                testData=testData(ix(1:1000),:);
                testClass=test_class(ix(1:1000),:);
                trainClass(trainClass~=1)=-1;
                trainClass(trainClass==2)=1;
                testClass(testClass~=1)=-1;
                testClass(testClass==2)=1;
                
                fprintf('Number of training data points %d-%d, class %d\n',size(train,1),size(train,2),size(train_class,1));
                fprintf('Number of test data points %d-%d\n',size(testData,1),size(testData,2));
                reportPoints=[nrSamples:batchSize:size(train,1)-batchSize]
                fprintf('Number of report points:%d',length(reportPoints))
                %we don't use validation here. We tune parameters on training data
                %(5-fold-crossvalidation)
                settings.XTrain=train;
                settings.YTrain=trainClass;
                settings.XTest=testData;
                settings.YTest=testClass;
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
                settings.ks=ks(kNN);
                settings.weightMode=WeightModes{ws};
                settings.neighbourMode=NeighborModes{ns};
                res=runExperiment(settings,method);
                results{r}=res;
                %save intermediate results just in case
                save(sprintf('%s/results.mat',outputPath),'results');
                %save intermediate results just in case
                avgAucs=zeros(1,length(reportPoints));
                realAvgAUCs=zeros(1,length(reportPoints));
                for i=1:r
                    avgAucs(i,:)=cell2mat(results{i}.selectedAUCs);
                    realAvgAUCs(i,:)=cell2mat(results{i}.AUCs);
                    allAucs(i,:)=cell2mat(results{i}.selectedAUCs);
                    allRealAucs(i,:)=cell2mat(results{i}.AUCs);
                    runTimes(i,:)=results{i}.runtime+results{i}.tuningTime;
                    processingTimes(i,:)=results{i}.processingTimes;
                end
                stdev=nanstd(allAucs);
                stdevReal=nanstd(allRealAucs);
                avgAucs=nanmean(avgAucs);
                realAvgAUCs=realAvgAUCs/nrRuns;
                avgRuntime=nanmean(runTimes);
                stdRuntime=nanstd(runTimes);
                save(sprintf('%s/auc.mat',outputPath),'avgAucs','realAvgAUCs','stdev','stdevReal','reportPoints','avgRuntime','stdRuntime','processingTimes');
            end
            avgAucs=zeros(1,length(reportPoints));
            realAvgAUCs=zeros(1,length(reportPoints));
            for i=1:nrRuns
                avgAucs(i,:)=cell2mat(results{i}.selectedAUCs);
                realAvgAUCs(i,:)=cell2mat(results{i}.AUCs);
                allAucs(i,:)=cell2mat(results{i}.selectedAUCs);
                allRealAucs(i,:)=cell2mat(results{i}.AUCs);
                runTimes(i,:)=results{i}.runtime+results{i}.tuningTime;
                processingTimes(i,:)=results{i}.processingTimes;
            end
            stdev=nanstd(allAucs,2);
            stdevReal=nanstd(allRealAucs,2);
            avgAucs=nanmean(avgAucs,2);
            realAvgAUCs=nanmean(realAvgAUCs,2);
            avgRuntime=nanmean(runTimes);
            stdRuntime=std(runTimes);
            
            save(sprintf('%s/auc.mat',outputPath),'avgAucs','realAvgAUCs','stdev','stdevReal','reportPoints','avgRuntime','stdRuntime','processingTimes');
            save(sprintf('%s/results.mat',outputPath),'results');
        end
    end
end