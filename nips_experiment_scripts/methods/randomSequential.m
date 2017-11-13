function [res]=randomSequential(trainFileID,trainOffsetIndices,formatting,delimiter,selectNum,batch,observationPoints,options,inferenceType)
starting_count=tic;
nrObsPoints=length(observationPoints);
res.selectedDataPoints=cell(1, nrObsPoints);
res.selectedLabels=cell(1, nrObsPoints);
res.selectedKernels=cell(1, nrObsPoints);
res.selectedDistances=cell(1, nrObsPoints);
res.selectedAUCs=cell(1, nrObsPoints);
res.times=zeros(1, nrObsPoints);
res.processingTimes=zeros(1, nrObsPoints);
res.selectedBetas=cell(1,nrObsPoints);
res.realBetas=cell(1,nrObsPoints);
res.percentageRemoved=cell(1,nrObsPoints);
res.trainAUCs=cell(1,nrObsPoints);

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
res.selectedKernels{point}=model.K;
res.selectedDistances{point}=model.D;
res.selectedDataPoints{point}=model.X;
res.selectedLabels{point}=model.Y;
res.times(point)=toc(starting_count);
res.reportPointIndex=point;
res.processingTimes(point)=toc(starting_count);
res.selectedAUCs{point}=current_area;
res.AUCs{point}=current_area;
res.trainAUCs{point}=aucTrain;
res.realBetas{point}=values;
res.selectedBetas{point}=values;
res.percentageRemoved{point}=0;
point=point+1;
pointerObs=selectNum;

while 1
    starting_count1=tic;
    %if pointerObs>=size(trainOffsetIndices,1)
    if pointerObs>=3000  %----------------------------------- REMOVE THIS!!!!
        break
    end
    ix=randperm(pointerObs+batch);
    indices=trainOffsetIndices(ix,:);
    %sample dataLimit datapoints from here
    oldModel=model;
    [XObserved,YObserved]=getDataInstancesSequential(trainFileID,formatting,delimiter,indices);
%     classes=unique(YObserved);
%     ix_up_class1=find(YObserved==classes(1));
%     ix_up_class2=find(YObserved==classes(2));
%     nr_samples1=ceil(selectNum/2);
%     nr_samples2=selectNum-nr_samples1;
%     if nr_samples1>size(ix_up_class1,1)
%         nr_samples1=size(ix_up_class1,1);
%         nr_samples2=selectNum-nr_samples1;
%     end
%     if nr_samples2>size(ix_up_class2,1)
%         nr_samples2=size(ix_up_class2,1);
%         nr_samples1=selectNum-nr_samples2;
%     end
% 
%     newModel.X=[XObserved(ix_up_class1(1:nr_samples2),:);XObserved(ix_up_class2(1:nr_samples2),:)];
%     newModel.Y=[YObserved(ix_up_class1(1:nr_samples2),:);YObserved(ix_up_class2(1:nr_samples2),:)];
    newModel.X=XObserved(1:selectNum,:);
    newModel.Y=YObserved(1:selectNum,:);
    [newModel,values]=MAED(newModel,selectNum,options);
    %keep the new model if it improves the auc
    fprintf('Model size for inference %d, Test class size %d\n',size(newModel.X,1),size(options.test,1));
    area=inferenceType(newModel.K,newModel.X,newModel.Y,options.test,options.test_class,options);
    areaTrain=-1;
    %areaTrain=inferenceType(newModel.K,newModel.X,newModel.Y,newModel.X,newModel.Y,options);
    area=max(area,1-area);
    %areaTrain=max(areaTrain,1-areaTrain);
    %area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options);
    if area<current_area
        model=oldModel;
    else
        current_area=area;
        model=newModel;
    end
    if point<=length(observationPoints) && pointerObs<=observationPoints(point)
        res.selectedDataPoints{point}=model.X;
        res.selectedLabels{point}=model.Y;
        res.selectedKernels{point}=model.K;
        res.times(point)=toc(starting_count);
        res.processingTimes(point)=toc(starting_count1);
        res.selectedAUCs{point}=current_area;
        res.percentageRemoved{point}=newModel.percentageRemoved;
        res.AUCs{point}=area;
        res.trainAUCs{point}=areaTrain;
        res.selectedBetas{point}=oldModel.betas;
        res.realBetas{point}=newModel.betas;
        res.reportPointIndex=point;
        %point=point+1;
        pointerObs=pointerObs+batch;
    end
    if pointerObs>=observationPoints(point)
        res.reportPointIndex=point;
        point=point+1;
        pointerObs=pointerObs+batch;
    end
end
end




