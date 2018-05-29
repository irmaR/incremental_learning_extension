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
%save current point
%current_area=inferenceType(model.X,model.Y,settings,settings,options)
area1=SRDASequential(model.X,model.Y,settings,options,settings.indicesOffsetValidation,settings.XTrainFileID);
area2=SVMSelectionSequential(model.X,model.Y,settings,options,settings.indicesOffsetValidation,settings.XTrainFileID);
current_area=nanmean([area1,area2]);
current_area=max(current_area,1-current_area);
%current_area=SRDASequential(model.X,model.Y,settings,options,settings.indicesOffsetTest,settings.XTestFileID);
aucTrain=-1;
%current_area=max(current_area,1-current_area);
aucTrain=max(aucTrain,1-aucTrain);
results.selectedKernels{point}=model.K;
results.selectedDistances{point}=model.D;
results.selectedDataPoints{point}=model.X;
results.selectedLabels{point}=model.Y;
results.times(point)=toc(starting_count);
results.reportPointIndex=point;
results.processingTimes(point)=toc(starting_count);
results.selectedAUCs{point}=current_area;
results.AUCs{point}=current_area;
results.trainAUCs{point}=aucTrain;
results.realBetas{point}=values;
results.selectedBetas{point}=values;
results.percentageRemoved{point}=0;
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
    %[XObserved,YObserved]=getDataInstancesSequential(settings.XTrainFileID,settings.formattingString,settings.delimiter,settings.indicesOffsetTrain(1:pointerObs+batch));
    newModel.X=[model.X;XNew];
    newModel.Y=[model.Y;YNew];
    %balanced
    classes=unique(newModel.Y);
    %determine how many samples to select from each class
    nr_samples1=ceil(settings.numSelectSamples/2);
    nr_samples2=settings.numSelectSamples-nr_samples1;
    ix=randperm(size(newModel.X,1));
    model.X=newModel.X(ix,:);
    model.Y=newModel.Y(ix,:);
    starting_count=tic;
    try
        ix_up_class1=find(model.Y==classes(1));
    catch
        ix_up_class1=[];
    end
    try
        ix_up_class2=find(model.Y==classes(2));
    catch
        ix_up_class2=[];
    end
    if nr_samples1>size(ix_up_class1,1)
        nr_samples1=size(ix_up_class1,1);
        nr_samples2=settings.numSelectSamples-nr_samples1;
    end
    
    if nr_samples2>size(ix_up_class2,1)
        nr_samples2=size(ix_up_class2,1);
        nr_samples1=settings.numSelectSamples-nr_samples2;
    end
    current_sample=[model.X(ix_up_class1(1:nr_samples1),:);model.X(ix_up_class2(1:nr_samples2),:)];
    current_labels=[model.Y(ix_up_class1(1:nr_samples1),:);model.Y(ix_up_class2(1:nr_samples2),:)];
    newModel.X=current_sample;
    newModel.Y=current_labels;
    %[a,~]=hist(newModel.Y,unique(newModel.Y));
    %[newModel,values]=MAED(newModel,settings.numSelectSamples,options);
    %keep the new model if it improves the auc
    %area=inferenceType(model.K,model.X,model.Y,settings,settings,options);
    area1=SRDASequential(newModel.X,newModel.Y,settings,options,settings.indicesOffsetValidation,settings.XTrainFileID);
    area2=SVMSelectionSequential(newModel.X,newModel.Y,settings,options,settings.indicesOffsetValidation,settings.XTrainFileID);
    areaSelection=nanmean([area1,area2]);
    areaSelection=max(areaSelection,1-areaSelection);
    fprintf('Area selection %f, current area %f\n',areaSelection,current_area);
    areaTrain=-1;
    if areaSelection<current_area
       model=oldModel;
    else
       current_area=areaSelection;
       model=newModel;
    end
    %get the test AUC given the current model
    area=SRDASequential(model.X,model.Y,settings,options,settings.indicesOffsetTest,settings.XTestFileID);
    areaSVM=SVMsequential(model.X,model.Y,settings,options);
    area=max(area,1-area)
    areaSVM=max(areaSVM,1-areaSVM)
    
    if point<=length(settings.reportPoints)
        results.selectedDataPoints{point}=model.X;
        results.selectedLabels{point}=model.Y;
        results.selectedKernels{point}=[];
        results.times(point)=toc(starting_count);
        results.processingTimes(point)=toc(starting_count1);
        results.selectedAUCs{point}=current_area;
        results.percentageRemoved{point}=-1;
        results.AUCs{point}=area;
        results.AUCSVM{point}=areaSVM;
        results.trainAUCs{point}=areaTrain;
        results.selectedBetas{point}=[];
        results.realBetas{point}=[];
        results.reportPointIndex=point;
        results.pointerObserved=pointerObs;
        results.TrainingIndices=settings.indicesOffsetTrain;
        results.processingTimes=results.processingTimes;
        results.selectionTimes=results.times;
        results.selectedBetas=results.selectedBetas;
        results.realBetas=results.realBetas;
        results.percentageRemoved=results.percentageRemoved;
        results.reportPoints=settings.reportPoints;
        results.reportPointIndex=results.reportPointIndex;
        save(sprintf('%s/results.mat',settings.outputPath),'results');
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




