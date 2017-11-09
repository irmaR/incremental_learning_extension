function [res]=MAEDBatch(trainX,trainY,selectNum,batch,dataLimit,options,observationPoints,balanced,inferenceType)
starting_count=tic;
res.selectedDataPoints=cell(1, length(observationPoints));
res.selectedLabels=cell(1, length(observationPoints));
res.selectedKernels=cell(1, length(observationPoints));
res.selectedDistances=cell(1, length(observationPoints));
res.selectedAUCs=cell(1, length(observationPoints));
res.times=zeros(1, length(observationPoints));
res.processingTimes=zeros(1, length(observationPoints));
res.selectedBetas=cell(1, length(observationPoints));
res.realBetas=cell(1, length(observationPoints));
res.percentageRemoved=cell(1, length(observationPoints));
res.trainAUCs=cell(1, length(observationPoints));

ix=randperm(size(trainX,1));
train_fea=trainX(ix,:);
train_class=trainY(ix,:);
model.X=train_fea(1:selectNum,:);
model.Y=train_class(1:selectNum,:);
point=1;

[model,values] = MAED(model,selectNum,options);
%save current point
current_area=inferenceType(model.K,model.X,model.Y,options.test,options.test_class,options);
aucTrain=inferenceType(model.K,model.X,model.Y,model.X,model.Y,options);
current_area=max(current_area,1-current_area);
aucTrain=max(aucTrain,1-aucTrain);

res.selectedKernels{point}=model.K;
res.selectedDistances{point}=model.D;
res.selectedDataPoints{point}=model.X;
res.selectedLabels{point}=model.Y;
res.times(point)=toc(starting_count);
res.processingTimes(point)=toc(starting_count);
res.selectedAUCs{point}=max(current_area,1-current_area);
res.trainAUCs{point}=max(aucTrain,1-aucTrain);
res.realBetas{point}=values;
res.AUCs{point}=current_area;
res.selectedBetas{point}=values;
res.percentageRemoved{point}=0;

point=point+1;

for j=0:batch:(size(train_fea,1)-selectNum-batch)
    starting_count1=tic;
    fprintf('Fetching %d - %d\t Batch %d\n',selectNum+j+1,selectNum+j+batch,j)
    %fprintf('Batch %d',j)
    %fprintf('iter %d',model_size+j-batch)
    %taking new points from the training pool
    XNew=trainX(selectNum+j+1:selectNum+j+batch,:);
    YNew=trainY(selectNum+j+1:selectNum+j+batch,:);
    oldModel=model;
    
    if balanced
        newModel=batchUpdateModelBalanced(model,options,XNew,YNew,selectNum,dataLimit);
    else
        newModel=batchUpdateModel(model,options,XNew,YNew,selectNum,dataLimit);
    end
    %keep the new model if it improves the auc
    area=inferenceType(newModel.K,newModel.X,newModel.Y,options.test,options.test_class,options);
    areaTrain=inferenceType(newModel.K,newModel.X,newModel.Y,newModel.X,newModel.Y,options);
    area=max(area,1-area);
    areaTrain=max(areaTrain,1-areaTrain);
    
    %area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options);
    if area<current_area
        model=oldModel;
    else
        current_area=area;
        model=newModel;
    end
    
    if point<=length(observationPoints) && selectNum+j<=observationPoints(point)
        %fprintf('reporting...')
        res.selectedDataPoints{point}=model.X;
        res.selectedLabels{point}=model.Y;
        res.times(point)=toc(starting_count);
        res.processingTimes(point)=toc(starting_count1);
        %fprintf('Point %d, Current Area %f\n',point,current_area)
        res.selectedAUCs{point}=current_area;
        res.percentageRemoved{point}=newModel.percentageRemoved;
        res.AUCs{point}=area;
        res.trainAUCs{point}=areaTrain;
        res.selectedBetas{point}=oldModel.betas;
        res.realBetas{point}=newModel.betas;
        point=point+1;
    end
    if point<=length(observationPoints) && selectNum+j>=observationPoints(point)
        point=point+1;
    end
end
end


