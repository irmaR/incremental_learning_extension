function [results]=MAEDBatchSequential(settings,options,inferenceType)
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
pointerObs=settings.numSelectSamples;
batch=settings.batchSize;
dataLimit=settings.dataLimit;
while 1
    starting_count1=tic;
    %if pointerObs>=size(trainOffsetIndices,1)    
    if pointerObs+batch>=size(settings.indicesOffsetTrain,1)  %----------------------------------- REMOVE THIS!!!!
        break
    end
    if pointerObs<=dataLimit
        [XNew,YNew]=getDataInstancesSequential(settings.XTrainFileID,settings.formattingString,settings.delimiter,settings.indicesOffsetTrain(1:pointerObs+batch));
        XObserved=[model.X;XNew];
        YObserved=[model.Y;YNew];
    else
        %sample dataLimit datapoints from here
        ix=randperm(pointerObs+batch);
        fprintf('Sampling from %d points',pointerObs+batch)
        indices=settings.indicesOffsetTrain(ix(1:dataLimit));
        [XObserved,YObserved]=getDataInstancesSequential(settings.XTrainFileID,settings.formattingString,settings.delimiter,indices);
    end
    fprintf('Size of new block %d\t, Reached %d\n',size(XObserved,1),pointerObs);
    oldModel=model;
    fprintf('Balanced %d\n',settings.balanced)
    if settings.balanced
        newModel=batchUpdateModelBalanced(model,options,XObserved,YObserved,settings.numSelectSamples);
    else
        fprintf('Updating the model\n')
        newModel=batchUpdateModel(model,options,XObserved,YObserved,settings.numSelectSamples);
    end
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
    if point<=length(settings.reportPoints) && pointerObs<=settings.reportPoints(point)
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
    if pointerObs>=settings.reportPoints(point)
        results.reportPointIndex=point;
        point=point+1;
        pointerObs=pointerObs+batch;
    end
end
end




