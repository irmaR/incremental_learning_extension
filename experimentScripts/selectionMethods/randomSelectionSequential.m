function [results]=randomSelectionSequential(settings,options,inferenceType)
starting_count=tic;
nrObsPoints=length(settings.reportPoints);
results.selectedDataPoints=cell(1, nrObsPoints);
results.selectedLabels=cell(1, nrObsPoints);
results.selectedKernels=cell(1, nrObsPoints);
results.selectedDistances=cell(1, nrObsPoints);
results.selectedAUCs=cell(1, nrObsPoints);
results.times=zeros(1, nrObsPoints);
results.processingTimes=zeros(1, nrObsPoints);
results.selectedBetas=cell(1,nrObsPoints);
results.realBetas=cell(1,nrObsPoints);
results.percentageRemoved=cell(1,nrObsPoints);
results.trainAUCs=cell(1,nrObsPoints);

%get first selectNum points from the file
indices=settings.indicesOffsetTrain(1:settings.numSelectSamples);
[model.X,model.Y]=getDataInstancesSequential(settings.XTrainFileID,settings.formattingString,settings.delimiter,indices);

point=1;
[model,values] = MAED(model,settings.numSelectSamples,options);
modelSRDA=model;
modelSRKDA=model;
modelSVM=model;
current_area1=SRDASequential(modelSRDA.X,modelSRDA.Y,settings,settings,settings.indicesOffsetValidation,settings.XTrainFileID);
current_area2=SVMSelectionSequential(modelSVM.X,modelSVM.Y,settings,settings,settings.indicesOffsetValidation,settings.XTrainFileID);
current_area3=SRKDASequential(modelSRKDA.X,modelSRKDA.Y,settings,settings,settings.indicesOffsetValidation,settings.XTrainFileID);
aucTrain=-1;
%current_area=max(current_area,1-current_area);
aucTrain=max(aucTrain,1-aucTrain);
results.selectedKernels{point}=model.K;
results.selectedDistances{point}=model.D;
results.selectedDataPoints{point}=model.X;
results.selectedLabels{point}=model.Y;
results.times(point)=toc(starting_count);
results.reportPointIndex=point;
results.AUCSRDA{point}=current_area1;
results.AUCSVM{point}=current_area2;
results.AUCSSRKDA{point}=current_area3;
results.processingTimes(point)=toc(starting_count);
point=point+1;
pointerObs=settings.numSelectSamples;
batch=settings.batchSize;
ix=randperm(pointerObs+batch);
indices=settings.indicesOffsetTrain(ix,:);
while 1
    starting_count1=tic;
    %indices=indices(1:settings.numSelectSamples);
    %sample dataLimit datapoints from here
    oldModel=model;
    fprintf('Slice %d - %d\n',pointerObs+1,pointerObs+batch)
    %[XObserved,YObserved]=getDataInstancesSequential(settings.XTrainFileID,settings.formattingString,settings.delimiter,indices);
    [XNew,YNew]=getDataInstancesSequential(settings.XTrainFileID,settings.formattingString,settings.delimiter,settings.indicesOffsetTrain(pointerObs+1:pointerObs+batch));
    oldModelSRDA=modelSRDA;
    oldModelSRKDA=modelSRKDA;
    oldModelSVM=modelSVM;
    newModel.X=[model.X;XNew];
    newModel.Y=[model.Y;YNew];
    %balanced
    classes=unique(newModel.Y);
    %determine how many samples to select from each class
    nr_samples1=ceil(settings.numSelectSamples/2);
    nr_samples2=settings.numSelectSamples-nr_samples1;
    ix=randperm(size(newModel.X,1));
    newModel.X=newModel.X(ix(1:settings.numSelectSamples),:);
    newModel.Y=newModel.Y(ix(1:settings.numSelectSamples),:);
%     starting_count=tic;
%     try
%         ix_up_class1=find(model.Y==classes(1));
%     catch
%         ix_up_class1=[];
%     end
%     try
%         ix_up_class2=find(model.Y==classes(2));
%     catch
%         ix_up_class2=[];
%     end
%     if nr_samples1>size(ix_up_class1,1)
%         nr_samples1=size(ix_up_class1,1);
%         nr_samples2=settings.numSelectSamples-nr_samples1;
%     end
%     
%     if nr_samples2>size(ix_up_class2,1)
%         nr_samples2=size(ix_up_class2,1);
%         nr_samples1=settings.numSelectSamples-nr_samples2;
%     end
%     current_sample=[model.X(ix_up_class1(1:nr_samples1),:);model.X(ix_up_class2(1:nr_samples2),:)];
%     current_labels=[model.Y(ix_up_class1(1:nr_samples1),:);model.Y(ix_up_class2(1:nr_samples2),:)];
%     newModel.X=current_sample;
%     newModel.Y=current_labels;

    
    time1=toc(starting_count);
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




