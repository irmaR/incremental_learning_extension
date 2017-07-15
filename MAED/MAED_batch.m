function [list_of_selected_data_points,list_of_selected_labels,list_of_selected_times,lists_of_processing_times,list_of_kernels,lists_of_dists,lists_of_areas]=MAED_batch(data,labels,model_size,batch_size,options,model_observation_points,data_limit,warping)
starting_count=tic;

%prepare data structures for reporting the results at each model
%observation point
list_of_selected_data_points=cell(1,length(model_observation_points));
list_of_selected_labels=cell(1,length(model_observation_points));
list_of_kernels=cell(1,length(model_observation_points));
lists_of_dists=cell(1,length(model_observation_points));
lists_of_areas=cell(1,length(model_observation_points));
list_of_selected_times=zeros(1,length(model_observation_points));
lists_of_processing_times=zeros(1,length(model_observation_points));

%shuffle data
ix=randperm(size(data,1));
[current_sample,current_labels,current_Dists,kernel,current_area]=initialize_sample(data(ix,:),labels(ix,:),model_size);
%store the current model in the list
point=1;
list_of_kernels{point}=kernel;
lists_of_dists{point}=current_Dists;
list_of_selected_data_points{point}=current_sample;
list_of_selected_labels{point}=current_labels;
list_of_selected_times(point)=toc(starting_count);
lists_of_processing_times(point)=toc(starting_count);
lists_of_areas{point}=current_area;


point=point+1;
%start the incremental loop

for j=0:batch_size:(size(data,1)-model_size-batch)
    starting_count1=tic;
    %fetch new batch size points from the data matrix
    new_points=data(model_size+j+1:model_size+j+batch_size,:);
    new_classes=labels(model_size+j+1:model_size+j+batch_size,:);
    
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
    %if the new sample does not improve the results, keep the previous
    %sample
    if area<current_area
        current_sample=old_sample;
        current_labels=old_labels;
        current_kernel=old_kernel;
        current_Dists=old_dists;
    else
        current_area=area;
    end
    
    
    %         case 'batch_bal'
    %             train_fea_incremental=[train_fea_incremental;new_points];
    %             train_fea_class_incremental=[train_fea_class_incremental;new_classes];
    %             new_points=[];
    %             new_classes=[];
    %             [current_sample,current_labels,ranking,kernel,current_Dists]=update_model_balance(options,model_size,ranking,values,current_sample,current_labels,new_points,new_classes,current_Dists,data_limit,warping,batch);
    
    %report the current model if the iteration corresponds to any
    %observation point
    if point<=length(model_observation_points) && model_size+j<=model_observation_points(point)
        %fprintf('reporting...')
        list_of_selected_data_points{point}=current_sample;
        list_of_selected_labels{point}=current_labels;
        if ~strcmp(experiment_type,'lssvm')
            list_of_kernels{point}=kernel;
            lists_of_dists{point}=current_Dists;
        end
        list_of_selected_times(point)=toc(starting_count);
        lists_of_processing_times(point)=toc(starting_count1);
        lists_of_areas{point}=current_area;
        %point=point+1;
    end
    if point<=length(model_observation_points) && model_size+j>=model_observation_points(point)
        point=point+1;
    end
end
end


function [sample,labels,D,kernel,AUC]=initialize_sample(data,labels,model_size)
%initial model: select first model_size points from the data
train_fea_incremental=train_fea(1:model_size,:);
train_fea_class_incremental=train_class(1:model_size,:);

[ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,size(train_fea_incremental,1),options,data_limit,warping);
current_sample=train_fea_incremental;
current_labels=train_fea_class_incremental;
current_Dists=current_D;
%save current point
current_area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options);
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
    if batch<nr_samples
        selected_samples=train_fea_incremental(ranking,:);
        indices_to_remove=ranking((size(selected_samples,1)+1)-size(new_points,1):end,:);
        selected_labels=train_fea_class_incremental(ranking,:);
        samples_updated=selected_samples(1:size(selected_samples,1)-size(new_points,1),:);
    else
        indices_to_remove=[];
        selected_samples=train_fea_incremental;
        selected_labels=train_fea_class_incremental;
        samples_updated=selected_samples;
    end
    
    
    [ranking,values,current_D,kernel,updated_sample,updated_class] = MAED_incremental(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,indices_to_remove,current_D,nr_samples,options,warping);
    %fprintf('Indices to remove')
    %fprintf('Kernel size %d',size(kernel,1))
    current_sample=updated_sample;
    current_labels=updated_class;
    %[ranking,values,current_D,kernel,current_sample,current_labels] = MAED_incremental_1(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,nr_samples,options);
end
end


function [current_sample,current_labels,ranking,kernel,current_D]=update_model_balance(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping,batch)
if size(new_points,1)==0
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
    
    [ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
    updated_class=train_fea_class_incremental(ranking,:);
    updated_sample=train_fea_incremental(ranking,:);
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
    [ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
else
    if batch<=nr_samples
        selected_samples=train_fea_incremental(ranking,:);
        indices_to_remove=ranking((size(selected_samples,1)+1)-size(new_points,1):end,:);
        selected_labels=train_fea_class_incremental(ranking,:);
        samples_updated=selected_samples(1:size(selected_samples,1)-size(new_points,1),:);
    else
        indices_to_remove=[];
        selected_samples=train_fea_incremental;
        selected_labels=train_fea_class_incremental;
        samples_updated=selected_samples;
    end
    classes=unique(train_fea_class_incremental);
    %     try
    %     ix1=find(train_fea_class_incremental==classes(1));
    %     catch
    %         ix1=[];
    %     end
    %     try
    %     ix2=find(train_fea_class_incremental==classes(2));
    %     catch
    %         ix2=[];
    %     end
    %determine how many samples to select from each class
    nr_samples1=ceil(nr_samples/2);
    nr_samples2=nr_samples-nr_samples1;
    indices_to_remove=[];
    [ranking,values,current_D,kernel,updated_sample,updated_class] = MAED_incremental(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,indices_to_remove,current_D,size(train_fea_incremental,1)+size(new_points,1),options,warping);
    
    try
        ix_up_class1=find(updated_class==classes(1));
    catch
        ix_up_class1=[];
    end
    try
        ix_up_class2=find(updated_class==classes(2));
    catch
        ix_up_class2=[];
    end
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
    [ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
    
    %[ranking,values,current_D,kernel,current_sample,current_labels] = MAED_incremental_1(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,nr_samples,options);
end
end


