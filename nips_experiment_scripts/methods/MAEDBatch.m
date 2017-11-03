function [res]=MAEDIncremental(trainX,trainY,selectNum,batch,dataLimit,options,observationPoints,balanced,inferenceType)
starting_count=tic;
res.selectedDataPoints=cell(1, length(observationPoints));
res.selectedLabels=cell(1, length(observationPoints));
res.selectedKernels=cell(1, length(observationPoints));
res.selectedDistances=cell(1, length(observationPoints));
res.selectedAUCs=cell(1, length(observationPoints));
res.times=zeros(1, length(observationPoints));
res.processingTimes=zeros(1, length(observationPoints));

ix=randperm(size(trainX,1));
train_fea=trainX(ix,:);
train_class=trainY(ix,:);
model.X=train_fea(1:selectNum,:);
model.Y=train_class(1:selectNum,:);
point=1;

[model,values] = MAED(model,selectNum,options);
%save current point
current_area=inferenceType(model.K,model.X,model.Y,options.test,options.test_class,options);
res.selectedKernels{point}=model.K;
res.selectedDistances{point}=model.D;
res.selectedDataPoints{point}=model.X;
res.selectedLabels{point}=model.Y;
res.times(point)=toc(starting_count);
res.processingTimes(point)=toc(starting_count);
res.selectedAUCs{point}=current_area;

point=point+1;

for j=0:batch:(size(train_fea,1)-selectNum-batch)
    starting_count1=tic;
    %fprintf('Fetching %d - %d\n',selectNum+j+1,selectNum+j+batch)
    %fprintf('Batch %d',j)
    %fprintf('iter %d',model_size+j-batch)
    %taking new points from the training pool
    XNew=trainX(selectNum+j+1:selectNum+j+batch,:);
    YNew=trainY(selectNum+j+1:selectNum+j+batch,:);
    oldModel=model;
    
    if balanced
        model=batchUpdateModelBalanced(model,options,XNew,YNew,selectNum,dataLimit);
    else
        model=batchUpdateModel(model,options,XNew,YNew,selectNum,dataLimit);
    end
    %keep the new model if it improves the auc
    area=inferenceType(model.K,model.X,model.Y,options.test,options.test_class,options);
    %area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options);
    if area<current_area
        model=oldModel;
    else
        current_area=area;
    end
    
    if point<=length(observationPoints) && selectNum+j<=observationPoints(point)
        %fprintf('reporting...')
        res.selectedDataPoints{point}=model.X;
        res.selectedLabels{point}=model.Y;
        res.times(point)=toc(starting_count);
        res.processingTimes(point)=toc(starting_count1);
        %fprintf('Point %d, Current Area %f\n',point,current_area)
        res.selectedAUCs{point}=current_area;
        point=point+1;
    end
    if point<=length(observationPoints) && selectNum+j>=observationPoints(point)
        point=point+1;
    end
end
end


