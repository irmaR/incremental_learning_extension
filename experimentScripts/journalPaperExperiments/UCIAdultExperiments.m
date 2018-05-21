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
                trainData=train(ix(1:ceil(size(ix,1)*2/3)),:);
                trainClass=train_class(ix(1:ceil(size(ix,1)*2/3),:));
                
                %standardize the training and test data
                [trainData,min_train,max_train]=standardizeX(trainData);
                testData=standardize(test,min_train,max_train);
                
                %this test dataset is pretty big so we will sample 1000 points in each
                %run
                ix=randperm(s,size(testData,1))';
                testData=testData(ix(1:1000),:);
                testClass=test_class(ix(1:1000),:);
                
                %I need validation data (a random subset of train data for
                %model selection)
                validation=trainData(1:2000,:);
                validationClass=trainClass(1:2000,:);
                trainData=trainData(2001:end,:);
                trainClass=trainClass(2001:end,:);
                
                fprintf('Number of training data points %d-%d, class %d\n',size(trainData,1),size(trainData,2),size(train_class,1));
                fprintf('Number of test data points %d-%d\n',size(testData,1),size(testData,2));
                reportPoints=[nrSamples:batchSize:size(trainData,1)-batchSize]
                fprintf('Number of report points:%d',length(reportPoints))
                
                settings.XTest=testData;
                settings.markSelPoints=1;
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
                settings.kernelType='RBF_kernel';
                settings.gamma=1;
                settings.warping=warping;
                settings.bLDA=balanced;
                settings.weightMode=WeightModes{ws};
                settings.neighbourMode=NeighborModes{ns};
                settings.ks=ks(kNN);
                settings.outputPath=outputPath;
                settings.reportPointIndex=1;
                settings.positiveClass=1;
                settings.classes=[1,2];
                res=runExperiment(settings,method)
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
    end
end