function [results]=FSLSSVM(settings,options,inferenceType)
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
%save current point
current_area=inferenceType(model.X,model.Y,options.test,options.test_class,options);
results.selectedDataPoints{point}=model.X;
results.selectedLabels{point}=model.Y;
results.times(point)=toc(starting_count);
results.processingTimes(point)=toc(starting_count);
results.selectedAUCs{point}=max(current_area,1-current_area);
results.AUCs{point}=current_area;

point=point+1;

for j=0:settings.batchSize:(size(settings.XTrain,1)-settings.numSelectSamples-settings.batchSize)
    starting_count1=tic;
    %fprintf('Fetching %d - %d\t Batch %d\n',settings.numSelectSamples+j+1,settings.numSelectSamples+j+settings.batchSize,j)
    XObserved=trainFea(1:settings.numSelectSamples+j+settings.batchSize,:);
    YObserved=trainClass(1:settings.numSelectSamples+j+settings.batchSize,:);
    nObs=size(XObserved,1);
    if size(XObserved,1)>=settings.dataLimit
        ix=randperm(size(XObserved,1));
        XObserved=XObserved(ix(1:settings.dataLimit),:);
        YObserved=YObserved(ix(1:settings.dataLimit),:);
    end
    fprintf('Number of observed (+ data limit) %d out of %d\n',size(XObserved,1),nObs)
    oldModel=model;
    Nc=settings.numSelectSamples;
    Xs=XObserved(1:Nc,:);
    Ys=YObserved(1:Nc,:);
    crit_old=-inf;
    for tel=1:200 %we are really doing them a favor, but it's so slow
        Xsp=Xs; Ysp=Ys;
        S=ceil(size(XObserved,1)*rand(1));
        Sc=ceil(Nc*rand(1));
        Xs(Sc,:) = XObserved(S,:);
        Ys(Sc,:) = YObserved(S);
        Ncc=Nc;
        crit = kentropy(Xs,options.kernel_type, options.kernel);
        if crit <= crit_old,
            crit = crit_old;
            Xs=Xsp;
            Ys=Ysp;
        else
            crit_old = crit;
        end
    end
    newModel.X=Xs;
    newModel.Y=Ys;
    areaSelection=inferenceType(newModel.X,newModel.Y,settings.validation,settings.validationClass,options);
    areaSelection=max(areaSelection,1-areaSelection);
    if areaSelection<current_area
        model=oldModel;
    else
        current_area=areaSelection;
        model=newModel;
    end
    if areaSelection<current_area
        model=oldModel;
    else
        current_area=areaSelection;
        model=newModel;
    end
    %get the test AUC given the current model
    area=inferenceType(model.X,model.Y,settings.XTest,settings.YTest,options);
    area=max(area,1-area);
    current_area=area;
    
    if point<=length(settings.reportPoints) && settings.numSelectSamples+j<=settings.reportPoints(point)
        results.selectedDataPoints{point}=model.X;
        results.selectedLabels{point}=model.Y;
        results.times(point)=toc(starting_count);
        results.processingTimes(point)=toc(starting_count1);
        results.selectedAUCs{point}=current_area;
        results.AUCs{point}=area;
        results.reportPointIndex=point;
        point=point+1;
    end
    if point<=length(settings.reportPoints) && settings.numSelectSamples+j>=settings.reportPoints(point)
        point=point+1;
    end
end


end