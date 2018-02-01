function [res]=MAEDBalancedIncremental(trainX,trainY,selectNum,batch,options,observationPoints,data_limit,experiment_type,warping,inferenceType)
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
if strcmp(experiment_type,'lssvm')
    current_area=0.5;
else
    [model,values] = MAED(model,selectNum,options);
    %save current point
    current_area=inferenceType(model.K,model.X,model.Y,options.test,options.test_class,options);
    res.selectedKernels{point}=model.K;
    res.selectedDistances{point}=model.D;
end
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
    %fprintf('Size of new points %d',size(new_points,1))
    %fprintf('Batch: %d',batch)
    
    %Batch or incremental experiment
    switch(experiment_type)
        case 'batch'
            train_fea_incremental=[train_fea_incremental;new_points];
            train_fea_class_incremental=[train_fea_class_incremental;new_classes];
            new_points=[];
            new_classes=[];
            old_sample=current_sample;
            old_labels=current_labels;
            old_kernel=kernel;
            old_dists=current_Dists;
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model(options,model_size,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_Dists,data_limit,warping,batch);
            area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options);
            if area<current_area
                current_sample=old_sample;
                current_labels=old_labels;
                current_kernel=old_kernel;
                current_Dists=old_dists;
            else
                current_area=area;
            end
        case 'incr'
            %fprintf('Train size %d\n',size(train_fea_incremental,1))
            %fprintf('New points %d\n',size(new_points,1))
            oldModel=model;
            [model] = MAEDRankIncremental(model,data(j:j+options.batchSize-1,:),labels(j:j+options.batchSize-1,:),options);
            %[current_sample,current_labels,ranking,kernel,current_Dists]=update_model(options,model_size,ranking,values,current_sample,current_labels,new_points,new_classes,current_Dists,data_limit,warping,batch);
            area=run_inference(model.K,model.X,model.Y,options.test,options.test_class,options);
            if area<current_area
                model=oldModel;
            else
                current_area=area;
            end
        case 'incr_bal'
            %fprintf('Train size %d\n',size(train_fea_incremental,1))
            %fprintf('New points %d\n',size(new_points,1))
            oldModel=model;
            model=updateModelBalanced(model,options,XNew,YNew,data_limit,selectNum);
            %fprintf('Kernel size:%d\n',size(kernel,1))
            %keep the new model if it improves the auc
            area=inferenceType(model.K,model.X,model.Y,options.test,options.test_class,options);
            %area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options);
            if area<current_area
                model=oldModel;
            else
                current_area=area;
            end
            
        case 'batch_bal'
            train_fea_incremental=[train_fea_incremental;new_points];
            train_fea_class_incremental=[train_fea_class_incremental;new_classes];
            new_points=[];
            new_classes=[];
            [current_sample,current_labels,ranking,kernel,current_Dists]=update_model_balance(options,model_size,ranking,values,current_sample,current_labels,new_points,new_classes,current_Dists,data_limit,warping,batch);
            
            
        case 'rnd'
            train_fea_incremental=[train_fea_incremental;new_points];
            train_fea_class_incremental=[train_fea_class_incremental;new_classes];
            %fprintf('Train size %d\n',size(train_fea_incremental,1))
            %[current_sample,current_labels,ranking,kernel,current_Dists]=update_model_random(options,model_size,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_Dists,data_limit,warping);
            old_sample=current_sample;
            old_labels=current_labels;
            old_kernel=kernel;
            old_dists=current_Dists;
            if options.bLDA
                [current_sample,current_labels,ranking,kernel,current_Dists]=update_model_random_balanced(options,model_size,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_Dists,data_limit,warping);
            else
                [current_sample,current_labels,ranking,kernel,current_Dists]=update_model_random(options,model_size,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_Dists,data_limit,warping);
            end
            area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options);
            if area<current_area
                current_sample=old_sample;
                current_labels=old_labels;
                kernel=old_kernel;
                current_Dists=old_dists;
            else
                current_area=area;
            end
            %fprintf('Kernel size:%d\n',size(kernel,1))
        case 'lssvm'
            train_fea_incremental=[train_fea_incremental;new_points];
            train_fea_class_incremental=[train_fea_class_incremental;new_classes];
            Nc=model_size;
            Xs=train_fea_incremental(1:Nc,:);
            Ys=train_fea_class_incremental(1:Nc,:);
            crit_old=-inf;
            for tel=1:500 %we are really doing them a favor, but it's so slow
                Xsp=Xs; Ysp=Ys;
                S=ceil(size(train_fea_incremental,1)*rand(1));
                Sc=ceil(Nc*rand(1));
                Xs(Sc,:) = train_fea_incremental(S,:);
                Ys(Sc,:) = train_fea_class_incremental(S);
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
            old_sample=current_sample;
            old_labels=current_labels;
            current_sample=Xs;
            current_labels=Ys;
            area=run_inference_lssvm(current_sample,train_fea,train_class,current_labels,options.test,options.test_class,options);
            if area<current_area
                current_sample=old_sample;
                current_labels=old_labels;
            else
                current_area=area;
            end
            % report selected points
    end
    %fprintf('Point %d, size observations %d',point,size(model_observation_points,2))
    %fprintf('Point %d, length %d,model s %d,model obs point %d, true? %d\n',point,length(model_observation_points),model_size+j,model_observation_points(point),point<=length(model_observation_points) && model_size+j<=model_observation_points(point))
    if point<=length(observationPoints) && selectNum+j<=observationPoints(point)
        %fprintf('reporting...')
        res.selectedDataPoints{point}=model.X;
        res.selectedLabels{point}=model.Y;
        if ~strcmp(experiment_type,'lssvm')
            res.selectedKernels{point}=model.K;
            res.selectedDistances{point}=model.D;
        end
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


function [current_sample,current_labels,ranking,kernel,current_D]=update_model_random(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping)
ix=randperm(size(train_fea_incremental,1));
train_fea_incremental=train_fea_incremental(ix(1:nr_samples),:);
train_fea_class_incremental=train_fea_class_incremental(ix(1:nr_samples),:);
[ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
%fprintf('Current kernel size: %d-%d',size(kernel,1),size(kernel,2))
current_sample=train_fea_incremental(ranking,:);
current_labels=train_fea_class_incremental(ranking,:);
kernel=kernel(ranking,ranking);
current_D=current_D(ranking,ranking);
%[ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
end

function [current_sample,current_labels,ranking,kernel,current_D]=update_model_random_balanced(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping)
ix=randperm(size(train_fea_incremental,1));
train_fea_incremental=train_fea_incremental(ix,:);
train_fea_class_incremental=train_fea_class_incremental(ix,:);
classes=unique(train_fea_class_incremental);
ix1=find(train_fea_class_incremental==classes(1));
ix2=find(train_fea_class_incremental==classes(2));

nr_samples1=ceil(nr_samples/2);
nr_samples2=nr_samples-nr_samples1;
if nr_samples1>size(ix1,1)
    nr_samples1=size(ix1,1);
    nr_samples2=nr_samples-nr_samples1;
end

if nr_samples2>size(ix2,1)
    nr_samples2=size(ix2,1);
    nr_samples1=nr_samples-nr_samples2;
end
train_fea_incremental=[train_fea_incremental(ix1(1:nr_samples1),:);train_fea_incremental(ix2(1:nr_samples2),:)];
train_fea_class_incremental=[train_fea_class_incremental(ix1(1:nr_samples1),:);train_fea_class_incremental(ix2(1:nr_samples2),:)];
[ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
%fprintf('Current kernel size: %d-%d',size(kernel,1),size(kernel,2))
current_sample=train_fea_incremental(ranking,:);
current_labels=train_fea_class_incremental(ranking,:);
%kernel=kernel(ranking,ranking);
%current_D=current_D(ranking,ranking);
[ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
end

function [current_sample,current_labels,ranking,kernel,current_D]=update_model(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping,batch)
if size(new_points,1)==0
    if size(train_fea_incremental,1)>=data_limit
        ix=randperm(size(train_fea_incremental,1));
        train_fea_incremental=train_fea_incremental(ix(1:data_limit),:);
        train_fea_class_incremental=train_fea_class_incremental(ix(1:data_limit),:);
    end
    [ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
    %fprintf('Current kernel size: %d-%d\n',size(kernel,1),size(kernel,2))
    current_sample=train_fea_incremental(ranking,:);
    current_labels=train_fea_class_incremental(ranking,:);
    %[ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
    kernel=kernel(ranking,ranking);
    current_D=current_D(ranking,ranking);
else
    selected_samples=train_fea_incremental(ranking,:);
    indices_to_remove=ranking((size(selected_samples,1)+1)-size(new_points,1):end,:);
    selected_labels=train_fea_class_incremental(ranking,:);
    samples_updated=selected_samples(1:size(selected_samples,1)-size(new_points,1),:);
end


[ranking,values,current_D,kernel,updated_sample,updated_class] = MAED_incremental(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,indices_to_remove,current_D,nr_samples,options,warping);
%fprintf('Indices to remove')
%fprintf('Kernel size %d',size(kernel,1))
current_sample=updated_sample;
current_labels=updated_class;
%[ranking,values,current_D,kernel,current_sample,current_labels] = MAED_incremental_1(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,nr_samples,options);
end



function [model]=updateModelBalanced(model,options,XNew,YNew,data_limit,numSamples)
if size(XNew,1)==0
    if size(train_fea_incremental,1)>=data_limit
        ix=randperm(size(train_fea_incremental,1));
        train_fea_incremental=train_fea_incremental(ix(1:data_limit),:);
        train_fea_class_incremental=train_fea_class_incremental(ix(1:data_limit),:);
    end
    %we assume that it's always binary problem, hence we split the data into
    %two classes
    classes=unique(train_fea_class_incremental);
    ix1=find(train_fea_class_incremental==classes(1));
    ix2=find(train_fea_class_incremental==classes(2));
    
    
    %determine how many samples to select from each class
    nr_samples1=ceil(nr_samples/2);
    nr_samples2=nr_samples-nr_samples1;
    train_1=train_fea_incremental(ix1,:);
    train_2=train_fea_incremental(ix2,:);
    class_1=train_fea_class_incremental(ix1,:);
    class_2=train_fea_class_incremental(ix2,:);
    [model,values] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options);
    %[ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
    updated_class=model.Y;
    updated_sample=model.X;
    ix_up_class1=find(updated_class==classes(1));
    ix_up_class2=find(updated_class==classes(2));
    
    if nr_samples1>size(ix_up_class1,1)
        nr_samples1=size(ix_up_class1,1);
        nr_samples2=nr_samples-nr_samples1;
    end
    
    if nr_samples2>size(ix_up_class2,1)
        nr_samples2=size(ix_up_class2,1);
        nr_samples1=nr_samples-nr_samples2;
    end
    
    current_sample=[updated_sample(ix_up_class1(1:nr_samples1),:);updated_sample(ix_up_class2(1:nr_samples2),:)];
    current_labels=[updated_class(ix_up_class1(1:nr_samples1),:);updated_class(ix_up_class2(1:nr_samples2),:)];
    [model,values]=MAED(current_sample,current_labels,numSamples,options);
else
    classes=unique(model.Y);
    %determine how many samples to select from each class
    nr_samples1=ceil(numSamples/2);
    nr_samples2=numSamples-nr_samples1;
    [model] = MAEDRankIncremental(model,XNew,YNew,numSamples,options);
    
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
        nr_samples2=numSamples-nr_samples1;
    end
    
    if nr_samples2>size(ix_up_class2,1)
        nr_samples2=size(ix_up_class2,1);
        nr_samples1=numSamples-nr_samples2;
    end
    
    current_sample=[model.X(ix_up_class1(1:nr_samples1),:);model.X(ix_up_class2(1:nr_samples2),:)];
    current_labels=[model.Y(ix_up_class1(1:nr_samples1),:);model.Y(ix_up_class2(1:nr_samples2),:)];
    [model]=MAED(model,numSamples,options);
end
end


