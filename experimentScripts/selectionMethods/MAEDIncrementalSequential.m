function [results]=MAEDIncrementalSequential(settings,options,inferenceType)
%function [res]=MAEDIncrementalSequential(trainFileID,trainOffsetIndices,formatting,delimiter,selectNum,batch,observationPoints,balanced,options,inferenceType)
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
sprintf('HALO')
size(settings.indicesOffsetTrain)
settings.numSelectSamples
indices=settings.indicesOffsetTrain(1:settings.numSelectSamples);
[model.X,model.Y]=getDataInstancesSequential(settings.XTrainFileID,settings.formattingString,settings.delimiter,indices);
point=1;
[model,values] = MAED(model,settings.numSelectSamples,options);
%save current point
current_area=inferenceType(model.K,model.X,model.Y,settings,settings,options);
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
batch=settings.batchSize;
pointerObs=settings.numSelectSamples;
while 1
    starting_count1=tic;
    %if pointerObs>=size(settings.indicesOffsetTrain,1)  %----------------------------------- REMOVE THIS!!!!
    %    batch=size(settings.indicesOffsetTrain,1)-pointerObs;
    %end
    if pointerObs>=size(settings.indicesOffsetTrain,1)
        break
    end
    fprintf('BATCH size: %d\n',batch);
    [XNew,YNew]=getDataInstancesSequential(settings.XTrainFileID,settings.formattingString,settings.delimiter,settings.indicesOffsetTrain(pointerObs+1:pointerObs+batch));
    fprintf('Size of new block %d\t, Reached %d\n',size(XNew,1),pointerObs);
    oldModel=model;
    if settings.balanced
        newModel=incrementalUpdateModelBalanced(model,options,XNew,YNew,settings.numSelectSamples);
    else
        newModel = MAEDRankIncremental(model,XNew,YNew,settings.numSelectSamples,options);
    end
    %keep the new model if it improves the auc
    sprintf('Running inference on model')
    area=inferenceType(newModel.K,newModel.X,newModel.Y,settings,settings,options);
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
    if area<current_area
        model=oldModel;
    else
        current_area=area;
        model=newModel;
    end
    if point<=length(settings.reportPoints)
        sprintf('HERE')
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
        if pointerObs+batch>=settings.reportPoints(point)
            results.reportPointIndex=point;
            point=point+1;
        end
    end
    
end
end


