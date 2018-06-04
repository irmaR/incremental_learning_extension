function [results]=MAEDIncrementalSequential(settings,inferenceType)
starting_count=tic;
nrObsPoints=length(settings.reportPoints);
results.selectedDataPoints=cell(1, nrObsPoints);
results.selectedLabels=cell(1, nrObsPoints);
results.selectedKernels=cell(1, nrObsPoints);
results.selectedDistances=cell(1, nrObsPoints);
results.selectedAUCs=cell(1, nrObsPoints);
results.AUCs=cell(1,nrObsPoints);
results.trainAUCs=cell(1, nrObsPoints);
results.times=zeros(1, nrObsPoints);
results.processingTimes=zeros(1, nrObsPoints);
results.selectedBetas=cell(1, nrObsPoints);
results.realBetas=cell(1, nrObsPoints);
results.percentageRemoved=cell(1,nrObsPoints);

%get first selectNum points from the file
indices=settings.indicesOffsetTrain(1:settings.numSelectSamples);
[model.X,model.Y]=getDataInstancesSequential(settings.XTrainFileID,settings.formattingString,settings.delimiter,indices);
point=1;
[model,values] = MAED(model,settings.numSelectSamples,settings);
%save current point
modelSRDA=model;
modelSRKDA=model;
modelSVM=model;
current_area1=SRDASequential(modelSRDA.X,modelSRDA.Y,settings,settings,settings.indicesOffsetValidation,settings.XTrainFileID);
current_area2=SVMSelectionSequential(modelSVM.X,modelSVM.Y,settings,settings,settings.indicesOffsetValidation,settings.XTrainFileID);
current_area3=SRKDASequential(modelSRKDA.X,modelSRKDA.Y,settings,settings,settings.indicesOffsetValidation,settings.XTrainFileID);
results.selectedKernels{point}=model.K;
results.selectedDistances{point}=model.D;
results.selectedDataPoints{point}=model.X;
results.selectedLabels{point}=model.Y;
results.times(point)=toc(starting_count);
results.reportPointIndex=point;
results.processingTimes(point)=toc(starting_count);
results.AUCSRDA{point}=current_area1;
results.AUCSVM{point}=current_area2;
results.AUCSSRKDA{point}=current_area3;
%point=point+1;
batch=settings.batchSize;
pointerObs=settings.numSelectSamples;
while 1
    starting_count1=tic;
    %if pointerObs>=size(settings.indicesOffsetTrain,1)
    %    break
    %end
    fprintf('Slice %d - %d\n',pointerObs+1,pointerObs+batch)
    settings.indicesOffsetTrain(pointerObs+1:pointerObs+batch);
    [XNew,YNew]=getDataInstancesSequential(settings.XTrainFileID,settings.formattingString,settings.delimiter,settings.indicesOffsetTrain(pointerObs+1:pointerObs+batch));
    %fprintf('Size of new block %d\t, Reached %d\n',size(XNew,1),pointerObs);
    oldModelSRDA=modelSRDA;
    oldModelSRKDA=modelSRKDA;
    oldModelSVM=modelSVM;
    if settings.balanced
        newModel=incrementalUpdateModelBalanced(model,settings,XNew,YNew,settings.numSelectSamples);
    else
        newModel = MAEDRankIncremental(model,XNew,YNew,settings.numSelectSamples,settings);
    end
    %keep the new model if it improves the au
    %run model selection on the validation data. Pick the model if it
    %improves the performance for each inference type
    time1=toc(starting_count);
    %areaSelection=log_reg_validation(newModel.K,newModel.X,newModel.Y,settings,settings,options);
    area1=SRDASequential(newModel.X,newModel.Y,settings,settings,settings.indicesOffsetValidation,settings.XTrainFileID);
    area2=SVMSelectionSequential(newModel.X,newModel.Y,settings,settings,settings.indicesOffsetValidation,settings.XTrainFileID);
    area3=SRKDASequential(newModel.X,newModel.Y,settings,settings,settings.indicesOffsetValidation,settings.XTrainFileID);
    fprintf('Area selection: SRDA=%0.2f, SRKDA=%0.2f, SVM=%0.2f\n',area1,area3,area2);
    area1=max(area1,1-area1);
    area2=max(area2,1-area2);
    area3=max(area3,1-area3);
    
    fprintf('SRDA: current area: %0.3f, new area: %0.3f\n',current_area1,area1);
    fprintf('SRKDA: current area: %0.3f, new area: %0.3f\n',current_area3,area3);
    fprintf('SVM: current area: %0.3f, new area: %0.3f\n',current_area2,area2);
    %choose preferred model SRDA
    if area1<current_area1
        modelSRDA=oldModelSRDA;
    else
        current_area1=area1;
        modelSRDA=newModel;
    end
    
    %choose preferred model SVM
    if area2<current_area2
        modelSVM=oldModelSVM;
    else
        current_area2=area2;
        modelSVM=newModel;
    end
    
    %choose preferred model SRKDA
    if area3<current_area3
        modelSRKDA=oldModelSRKDA;
        fprintf('Keeping the old model\n');
    else
        current_area3=area3;
        modelSRKDA=newModel;
    end
    

    %get the test AUC given the current model
    areaSRDA=SRDASequential(modelSRDA.X,modelSRDA.Y,settings,settings,settings.indicesOffsetTest,settings.XTestFileID);
    areaSRKDA=SRKDASequential(modelSRKDA.X,modelSRKDA.Y,settings,settings,settings.indicesOffsetTest,settings.XTestFileID);
    areaSVM=SVMsequential(modelSVM.X,modelSVM.Y,settings,settings);
    areaSRDA=max(areaSRDA,1-areaSRDA)
    areaSRKDA=max(areaSRKDA,1-areaSRKDA)
    areaSVM=max(areaSVM,1-areaSVM)

    if point<=length(settings.reportPoints)
        results.selectedDataPoints{point}=model.X;
        results.selectedLabels{point}=model.Y;
        results.selectedKernels{point}=model.K;
        results.times(point)=time1;
        results.processingTimes(point)=toc(starting_count1);        
        results.AUCSRDA{point}=areaSRDA;
        results.AUCSVM{point}=areaSVM;
        results.AUCSSRKDA{point}=areaSRKDA;
        results.reportPointIndex=point;
        results.pointerObserved=pointerObs;
        results.TrainingIndices=settings.indicesOffsetTrain;
        results.processingTimes=results.processingTimes;
        results.selectionTimes=results.times;
        results.reportPoints=settings.reportPoints;
        results.reportPointIndex=results.reportPointIndex;
        save(sprintf('%s/results.mat',settings.outputPath),'results');
        %fprintf('Reported at point %d',point)
        pointerObs=pointerObs+batch;
    end
    if point+1<=length(settings.reportPoints)
        if pointerObs+batch>=settings.reportPoints(point+1)
            results.reportPointIndex=point;
            point=point+1;
        end
    else
        break;
    end
end
end


