function [results]=randomSelection(settings,options,inferenceType)
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

ix=randperm(size(settings.XTrain,1));
trainFea=settings.XTrain(ix,:);
trainClass=settings.YTrain(ix,:);
model.X=trainFea(1:settings.numSelectSamples,:);
model.Y=trainClass(1:settings.numSelectSamples,:);
point=1;

[model,values] = MAED(model,settings.numSelectSamples,options);
%save current point
current_area=inferenceType(model.K,model.X,model.Y,options.test,options.test_class,options);
aucTrain=inferenceType(model.K,model.X,model.Y,model.X,model.Y,options);
current_area=max(current_area,1-current_area);
aucTrain=max(aucTrain,1-aucTrain);

results.selectedKernels{point}=model.K;
results.selectedDistances{point}=model.D;
results.selectedDataPoints{point}=model.X;
results.selectedLabels{point}=model.Y;
results.times(point)=toc(starting_count);
results.processingTimes(point)=toc(starting_count);
results.selectedAUCs{point}=max(current_area,1-current_area);
results.trainAUCs{point}=max(aucTrain,1-aucTrain);
results.realBetas{point}=values;
results.AUCs{point}=current_area;
results.selectedBetas{point}=values;
results.percentageRemoved{point}=0;

point=point+1;

for j=0:settings.batchSize:(size(settings.XTrain,1)-settings.numSelectSamples-settings.batchSize)
    starting_count1=tic;
    %fprintf('Fetching %d - %d\t Batch %d\n',settings.numSelectSamples+j+1,settings.numSelectSamples+j+settings.batchSize,j)
    XObserved=trainFea(1:settings.numSelectSamples+j+settings.batchSize,:);
    YObserved=trainClass(1:settings.numSelectSamples+j+settings.batchSize,:);
    ix=randperm(size(XObserved,1));
    X=XObserved(ix(1:settings.numSelectSamples),:);
    Y=YObserved(ix(1:settings.numSelectSamples),:);
    oldModel=model;
    newModel.X=X;
    newModel.Y=Y;
    [newModel,~] = MAED(newModel,settings.numSelectSamples,options);
    %keep the new model if it improves the auc
    area=inferenceType(newModel.K,newModel.X,newModel.Y,options.test,options.test_class,options);
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
    
    if point<=length(settings.reportPoints) && settings.numSelectSamples+j<=settings.reportPoints(point)
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
        point=point+1;
    end
    if point<=length(settings.reportPoints) && settings.numSelectSamples+j>=settings.reportPoints(point)
        point=point+1;
    end
end
end

