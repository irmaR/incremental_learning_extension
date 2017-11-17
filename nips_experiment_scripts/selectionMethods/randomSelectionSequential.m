function [results]=randomSelectionSequential(trainFileID,trainOffsetIndices,formatting,delimiter,selectNum,batch,observationPoints,options,inferenceType)
starting_count=tic;
nrObsPoints=length(observationPoints);
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
indices=trainOffsetIndices(1:selectNum);
[model.X,model.Y]=getDataInstancesSequential(trainFileID,formatting,delimiter,indices);
point=1;
[model,values] = MAED(model,selectNum,options);
%save current point
current_area=inferenceType(model.K,model.X,model.Y,options.test,options.test_class,options);
aucTrain=-1;
current_area=max(current_area,1-current_area);
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
pointerObs=selectNum;

while 1
    starting_count1=tic;
    if pointerObs+batch>=size(trainOffsetIndices,1)
        break
    end
    ix=randperm(pointerObs+batch);
    indices=trainOffsetIndices(ix,:);
    indices=indices(1:selectNum);
    %sample dataLimit datapoints from here
    oldModel=model;
    [XObserved,YObserved]=getDataInstancesSequential(trainFileID,formatting,delimiter,indices);
    newModel.X=XObserved;
    newModel.Y=YObserved;
    [newModel,values]=MAED(newModel,selectNum,options);
    %keep the new model if it improves the auc
    fprintf('Model size for inference %d, Test class size %d\n',size(newModel.X,1),size(options.test,1));
    area=inferenceType(newModel.K,newModel.X,newModel.Y,options.test,options.test_class,options);
    areaTrain=-1;
    area=max(area,1-area);
    if area<current_area
        model=oldModel;
    else
        current_area=area;
        model=newModel;
    end
    if point<=length(observationPoints) && pointerObs<=observationPoints(point)
        results.selectedDataPoints{point}=model.X;
        results.selectedLabels{point}=model.Y;
        results.selectedKernels{point}=model.K;
        results.times(point)=toc(starting_count);
        results.processingTimes(point)=toc(starting_count1);
        results.selectedAUCs{point}=current_area;
        results.percentageRemoved{point}=newModel.percentageRemoved;
        results.AUCs{point}=area;
        results.trainAUCs{point}=areaTrain;
        results.selectedBetas{point}=oldModel.betas;
        results.realBetas{point}=newModel.betas;
        results.reportPointIndex=point;
        pointerObs=pointerObs+batch;
    end
    if pointerObs>=observationPoints(point)
        results.reportPointIndex=point;
        point=point+1;
        pointerObs=pointerObs+batch;
    end
end
end




