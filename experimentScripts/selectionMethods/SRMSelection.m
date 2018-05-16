function [results]=SRMSelection(settings,options,inferenceType)
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

trainFea=settings.XTrain;
trainClass=settings.YTrain;
model.X=trainFea(1:settings.numSelectSamples,:);
model.Y=trainClass(1:settings.numSelectSamples,:);
model.K = constructKernel(full(model.X), [], options);
point=1;
%save current point
current_area=inferenceType(model.K,model.X,model.Y,options.test,options.test_class,options);
[areaSRKDA,areaSRDA,areaDT,areaRidge,areaSVM]=run_all_inferences(model,settings.XTest,settings.YTest,options)
results.selectedDataPoints{point}=model.X;
results.selectedLabels{point}=model.Y;
results.times(point)=toc(starting_count);
results.processingTimes(point)=toc(starting_count);
results.selectedAUCs{point}=max(current_area,1-current_area);
results.AUCs{point}=current_area;
results.SRKDAAucs{point}=areaSRKDA;
results.DTAUCs{point}=areaDT;
results.SVMAUCs{point}=areaSVM;
results.RidgeAUCs{point}=areaRidge;
results.SRDAAUC{point}=areaSRDA;

point=point+1;
numSelectSamples=settings.numSelectSamples;
for j=0:settings.batchSize:(size(settings.XTrain,1)-settings.numSelectSamples-settings.batchSize)
    starting_count1=tic;
    %fprintf('Fetching %d - %d\t Batch %d\n',settings.numSelectSamples+j+1,settings.numSelectSamples+j+settings.batchSize,j)
    XObserved=trainFea(1:numSelectSamples+j+settings.batchSize,:);
    YObserved=trainClass(1:numSelectSamples+j+settings.batchSize,:);
    nObs=size(XObserved,1);
    if size(XObserved,1)>=settings.dataLimit
        ix=randperm(size(XObserved,1));
        XObserved=XObserved(ix(1:settings.dataLimit),:);
        YObserved=YObserved(ix(1:settings.dataLimit),:);
    end
    fprintf('Number of observed (+ data limit) %d out of %d\n',size(XObserved,1),nObs)
    %split observed data into two classes
    %classes=unique(YObserved);
    %nr_samples1=ceil(settings.numSelectSamples/2);
    %nr_samples2=settings.numSelectSamples-nr_samples1;
    %split data into two classes
    %ix_up_class1=find(YObserved==classes(1));
    %ix_up_class2=find(YObserved==classes(2));
    %if nr_samples1>size(ix_up_class1,1)
    %    nr_samples1=size(ix_up_class1,1);
    %    nr_samples2=numSamples-nr_samples1;
    %end
    
    %if nr_samples2>size(ix_up_class2,1)
    %    nr_samples2=size(ix_up_class2,1);
    %    nr_samples1=numSamples-nr_samples2;
    %end
    %XObservedClass1=XObserved(ix_up_class1,:);
    %XObservedClass2=XObserved(ix_up_class2,:);
    %YObservedClass1=YObserved(ix_up_class1,:);
    %YObservedClass2=YObserved(ix_up_class2,:);
    
    %find representatives for class1
    %[XClass1,YClass1]=SRMSBalanced(XObservedClass1,YObservedClass1,nr_samples1,options);
    %[XClass2,YClass2]=SRMSBalanced(XObservedClass2,YObservedClass2,nr_samples2,options);
    oldModel=model;
    %Nc=settings.numSelectSamples;
    [XClass1,YClass1]=SRMSBalanced(XObserved,YObserved,settings.numSelectSamples,options);
    startInferenceTime=tic;
    numSelectSamples=size(XClass1,1)
    newModel.X=full(XClass1);
    newModel.Y=YClass1;
    newModel.K = constructKernel(newModel.X, [], options);
    [areaSRKDA,areaSRDA,areaDT,areaRidge,areaSVM]=run_all_inferences(newModel,settings.validation,settings.validationClass,options);
    areaSelection=nanmean([areaSRKDA,areaSRDA,areaDT,areaRidge,areaSVM]);
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
    %model.K = constructKernel(full(model.X), [], options);
    %area=inferenceType(model.X,model.Y,settings.XTest,settings.YTest,options);
    %area=max(area,1-area);
    [areaSRKDA,areaSRDA,areaDT,areaRidge,areaSVM]=run_all_inferences(model,settings.XTest,settings.YTest,options);
    current_area=areaSelection;
    inferenceTime=toc(startInferenceTime);
    if point<=length(settings.reportPoints) && settings.numSelectSamples+j<=settings.reportPoints(point)
        results.selectedDataPoints{point}=model.X;
        results.selectedLabels{point}=model.Y;
        results.times(point)=toc(starting_count);
        results.processingTimes(point)=toc(starting_count)-inferenceTime;
        results.selectedAUCs{point}=current_area;
        results.AUCs{point}=areaSRKDA;
        results.selectedSamples=numSelectSamples;
        results.reportPointIndex=point;
        results.SRKDAAucs{point}=areaSRKDA;
        results.DTAUCs{point}=areaDT;
        results.SVMAUCs{point}=areaSVM;
        results.RidgeAUCs{point}=areaRidge;
        results.SRDAAUC{point}=areaSRDA;
        point=point+1;
    end
    if point<=length(settings.reportPoints) && settings.numSelectSamples+j>=settings.reportPoints(point)
        point=point+1;
    end
end




end