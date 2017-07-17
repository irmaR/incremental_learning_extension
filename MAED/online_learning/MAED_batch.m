function [experiment_info,current_sample,current_labels,kernel]=MAED_batch(data,labels,numSample,batch_size,options,data_limit)
% MAED_incremental: Online Batch Manifold Adaptive Experimental Design
%     sampleList = MAED(fea,selectNum,options)
% Input:
%   data               - Data matrix MXN, where M is the number of data
%                          points and N is then number of features
%   labels             - Labels for data (Mx1)
%   numSample          - The size of the fixed-size model
%   options            - Struct value in Matlab. The fields in options
%                               that can be set:
%
%   observation_points -  an array consisting of desired observation
%                         snapshots of the algorithm. For
%                         example, model_observation_points=[50,100] will record
%                         results obtained after observing 50 and 100 points
%                         respectively.
%   test_data          -  if one wants to retain only the best model
%                         encountered during the incremental learning, a
%                         test data structure of the following format
%                         should be added:
%                          test_data.data
%                          test_data.labels
%
%   W       -  Affinity matrix. You can either call
%                                 "constructW" to construct the W, or
%                                 construct it by yourself.
%                                 If 'W' is not provided and 'ReguBeta'>0 ,
%                                 MAED will build a k-NN graph with Heat kernel
%                                 weight, where 'k' is a prameter.
%
%   k       -  The parameter for k-NN graph (Default is 5)
%                                 If 'W' is provided, this parameter will be
%                                 ignored.
%
%   ReguBeta    -  regularization parameter for manifold
%                                 adaptive kernel.
%
%   ReguAlpha   -  ridge regularization paramter. Default 0.01
%Output:
%
%        experiment info (if specified throug observation points flag in options)     - The index of the sample which should be labeled.
%        current sample     - Final data points retained by the incremental
%                             approach
%        current_kernel     - Final kernel model

experiment_info={}; %will store results at different observation snapshots
starting_count=tic;
%shuffle data
ix=randperm(size(data,1));
data=data(ix,:);
labels=labels(ix,:);
[current_sample,current_labels,current_D,kernel]=initialize_sample(options,data,labels,numSample);

point=1; %observation point counter
%check if we record experiment info at observation points kept in option.
%If yes, real model observation points in options and save this current
%model as the starting point
if isfield(options,'model_observation_points')
    experiment_info.list_of_kernels{point}=kernel;
    experiment_info.lists_of_dists{point}=current_D;
    experiment_info.list_of_selected_data_points{point}=current_sample;
    experiment_info.list_of_selected_labels{point}=current_labels;
    experiment_info.list_of_selected_times(point)=toc(starting_count);
    experiment_info.lists_of_processing_times(point)=toc(starting_count);
    if isfield(options,'test_data')
        current_area=run_inference(kernel,sample,labels,options.test_data.data,options.test_data.labels,options);
    end
    experiment_info.lists_of_areas{point}=current_area;
    point=point+1;
end

point=point+1;
%start the incremental loop
for j=0:batch_size:(size(data,1)-numSample-batch_size)
    starting_count1=tic;
    %expand data with new points
    train_fea_incremental=data(1:numSample+j+batch_size,:);
    train_fea_class_incremental=labels(1:numSample+j+batch_size,:);
    %if data limit reached use only a randomly selected sample of
    %data_limit points
    if size(train_fea_incremental,1)>=data_limit
        ix=randperm(size(train_fea_incremental,1));
        train_fea_incremental=train_fea_incremental(ix(1:data_limit),:);
        train_fea_class_incremental=train_fea_class_incremental(ix(1:data_limit),:);
    end
    fprintf('Size train %d\n',size(train_fea_incremental,1))
    old_sample=current_sample;
    old_labels=current_labels;
    old_kernel=kernel;
    [ranking,kernel] = MAED_batch_ranking(train_fea_incremental,train_fea_class_incremental,numSample,options);
    current_sample=train_fea_incremental(ranking,:);
    current_labels=train_fea_class_incremental(ranking,:);
    %if the new sample does not improve the results, keep the previous
    %sample
    if isfield(options,'test_data')
        area=run_inference(kernel,current_sample,current_labels,options.test,options.test_class,options);
        if area<current_area
            current_sample=old_sample;
            current_labels=old_labels;
            kernel=old_kernel;
        else
            current_area=area;
        end
    end
    %report the current model if the iteration corresponds to any
    %observation point
    if isfield(options,'model_observation_points')
        if point<=length(model_observation_points) && options.model_size+j<=model_observation_points(point)
            %fprintf('reporting...')
            experiment_info.list_of_selected_data_points{point}=current_sample;
            experiment_info.list_of_selected_labels{point}=current_labels;
            experiment_info.list_of_kernels{point}=current_kernel;
            experiment_info.list_of_selected_times(point)=toc(starting_count);
            experiment_info.lists_of_processing_times(point)=toc(starting_count1);
            experiment_info.lists_of_areas{point}=current_area;
        end
        if point<=length(model_observation_points) && options.model_size+j>=model_observation_points(point)
            point=point+1;
        end
    end
end

    function [sample,labels,ranking,kernel]=initialize_sample(options,data,labels,model_size)
        %initial model: select first model_size points from the data
        init_fea=data(1:model_size,:);
        init_labels=labels(1:model_size,:);
        [ranking,kernel] = MAED_batch_ranking(init_fea,init_labels,size(init_fea,1),options);
        sample=init_fea;
        labels=init_labels;
    end

    function [sampleList,K] = MAED_batch_ranking(fea,labels,selectNum,options)
        %Reference:
        %
        %   [1] Deng Cai and Xiaofei He, "Manifold Adaptive Experimental Design for
        %   Text Categorization", IEEE Transactions on Knowledge and Data
        %   Engineering, vol. 24, no. 4, pp. 707-719, 2012.
        %
        %   version 2.0 --Jan/2012
        %   version 1.0 --Aug/2008
        %
        %   Written by Deng Cai (dengcai AT gmail.com)
        nSmp = size(fea,1);
        splitLabel = false(nSmp,1);
        if isfield(options,'splitLabel')
            splitLabel = options.splitLabel;
        end
        
        [K,Dist,options] = constructKernel(fea,[],options);
        if isfield(options,'warping') && options.warping
            options.gnd=labels;
        end
        if isfield(options,'ReguBeta') && options.ReguBeta > 0
            if isfield(options,'W')
                W = options.W;
            else
                if isfield(options,'k')
                    Woptions.k = options.k;
                else
                    Woptions.k = 5;
                end
                
                tmpD = Dist;
                Woptions.t = mean(mean(tmpD));
                if isfield(options,'gnd')
                    Woptions.WeightMode = 'HeatKernel';
                    Woptions.NeighborMode='Supervised';
                    Woptions.bLDA=options.bLDA;
                    Woptions.gnd=options.gnd;
                end
                W = constructW(fea,Woptions);
            end
            D = full(sum(W,2));
            L = spdiags(D,0,nSmp,nSmp)-W;
            K=(speye(size(K,1))+options.ReguBeta*K*L)\K;
            K = max(K,K');
        end
        
        if ~isfield(options,'Method')
            options.Method = 'Seq';
        end
        ReguAlpha = 0.01;
        if isfield(options,'ReguAlpha')
            ReguAlpha = options.ReguAlpha;
        end
        switch lower(options.Method)
            case {lower('Seq')}
                [sampleList,values] = MAEDseq(K,selectNum,splitLabel,ReguAlpha);
            otherwise
                error('Optimization method does not exist!');
        end
    end
end

% function [current_sample,current_labels,ranking,kernel,current_D]=update_model_random(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping)
% ix=randperm(size(train_fea_incremental,1));
% train_fea_incremental=train_fea_incremental(ix(1:nr_samples),:);
% train_fea_class_incremental=train_fea_class_incremental(ix(1:nr_samples),:);
% [ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
% %fprintf('Current kernel size: %d-%d',size(kernel,1),size(kernel,2))
% current_sample=train_fea_incremental(ranking,:);
% current_labels=train_fea_class_incremental(ranking,:);
% kernel=kernel(ranking,ranking);
% current_D=current_D(ranking,ranking);
% %[ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
% end

% function [current_sample,current_labels,ranking,kernel,current_D]=update_model_random_balanced(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping)
% ix=randperm(size(train_fea_incremental,1));
% train_fea_incremental=train_fea_incremental(ix,:);
% train_fea_class_incremental=train_fea_class_incremental(ix,:);
% classes=unique(train_fea_class_incremental);
% ix1=find(train_fea_class_incremental==classes(1));
% ix2=find(train_fea_class_incremental==classes(2));
%
% nr_samples1=ceil(nr_samples/2);
% nr_samples2=nr_samples-nr_samples1;
% if nr_samples1>size(ix1,1)
%     nr_samples1=size(ix1,1);
%     nr_samples2=nr_samples-nr_samples1;
% end
%
% if nr_samples2>size(ix2,1)
%     nr_samples2=size(ix2,1);
%     nr_samples1=nr_samples-nr_samples2;
% end
%
% train_fea_incremental=[train_fea_incremental(ix1(1:nr_samples1),:);train_fea_incremental(ix2(1:nr_samples2),:)];
% train_fea_class_incremental=[train_fea_class_incremental(ix1(1:nr_samples1),:);train_fea_class_incremental(ix2(1:nr_samples2),:)];
% [ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
% %fprintf('Current kernel size: %d-%d',size(kernel,1),size(kernel,2))
% current_sample=train_fea_incremental(ranking,:);
% current_labels=train_fea_class_incremental(ranking,:);
% %kernel=kernel(ranking,ranking);
% %current_D=current_D(ranking,ranking);
% [ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
% end

% function [current_sample,current_labels,ranking,kernel,current_D]=update_model(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping,batch)
% if size(new_points,1)==0
%     if size(train_fea_incremental,1)>=data_limit
%         ix=randperm(size(train_fea_incremental,1));
%         train_fea_incremental=train_fea_incremental(ix(1:data_limit),:);
%         train_fea_class_incremental=train_fea_class_incremental(ix(1:data_limit),:);
%     end
%     [ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
%     %fprintf('Current kernel size: %d-%d\n',size(kernel,1),size(kernel,2))
%     current_sample=train_fea_incremental(ranking,:);
%     current_labels=train_fea_class_incremental(ranking,:);
%     %[ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
%     kernel=kernel(ranking,ranking);
%     current_D=current_D(ranking,ranking);
%
% else
%     if batch<nr_samples
%         selected_samples=train_fea_incremental(ranking,:);
%         indices_to_remove=ranking((size(selected_samples,1)+1)-size(new_points,1):end,:);
%         selected_labels=train_fea_class_incremental(ranking,:);
%         samples_updated=selected_samples(1:size(selected_samples,1)-size(new_points,1),:);
%     else
%         indices_to_remove=[];
%         selected_samples=train_fea_incremental;
%         selected_labels=train_fea_class_incremental;
%         samples_updated=selected_samples;
%     end
%
%
%     [ranking,values,current_D,kernel,updated_sample,updated_class] = MAED_incremental(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,indices_to_remove,current_D,nr_samples,options,warping);
%     %fprintf('Indices to remove')
%     %fprintf('Kernel size %d',size(kernel,1))
%     current_sample=updated_sample;
%     current_labels=updated_class;
%     %[ranking,values,current_D,kernel,current_sample,current_labels] = MAED_incremental_1(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,nr_samples,options);
% end
% end


% function [current_sample,current_labels,ranking,kernel,current_D]=update_model_balance(options,nr_samples,ranking,values,train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,data_limit,warping,batch)
% if size(new_points,1)==0
%     if size(train_fea_incremental,1)>=data_limit
%         ix=randperm(size(train_fea_incremental,1));
%         train_fea_incremental=train_fea_incremental(ix(1:data_limit),:);
%         train_fea_class_incremental=train_fea_class_incremental(ix(1:data_limit),:);
%     end
%     %we assume that it's always binary problem, hence we split the data into
%     %two classes
%     classes=unique(train_fea_class_incremental);
%     ix1=find(train_fea_class_incremental==classes(1));
%     ix2=find(train_fea_class_incremental==classes(2));
%
%
%     %determine how many samples to select from each class
%     nr_samples1=ceil(nr_samples/2);
%     nr_samples2=nr_samples-nr_samples1;
%
%
%
%     train_1=train_fea_incremental(ix1,:);
%     train_2=train_fea_incremental(ix2,:);
%     class_1=train_fea_class_incremental(ix1,:);
%     class_2=train_fea_class_incremental(ix2,:);
%
%     [ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
%     updated_class=train_fea_class_incremental(ranking,:);
%     updated_sample=train_fea_incremental(ranking,:);
%     ix_up_class1=find(updated_class==classes(1));
%     ix_up_class2=find(updated_class==classes(2));
%
%     if nr_samples1>size(ix_up_class1,1)
%         nr_samples1=size(ix_up_class1,1);
%         nr_samples2=nr_samples-nr_samples1;
%     end
%
%     if nr_samples2>size(ix_up_class2,1)
%         nr_samples2=size(ix_up_class2,1);
%         nr_samples1=nr_samples-nr_samples2;
%     end
%
%     current_sample=[updated_sample(ix_up_class1(1:nr_samples1),:);updated_sample(ix_up_class2(1:nr_samples2),:)];
%     current_labels=[updated_class(ix_up_class1(1:nr_samples1),:);updated_class(ix_up_class2(1:nr_samples2),:)];
%     [ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
% else
%     if batch<=nr_samples
%         selected_samples=train_fea_incremental(ranking,:);
%         indices_to_remove=ranking((size(selected_samples,1)+1)-size(new_points,1):end,:);
%         selected_labels=train_fea_class_incremental(ranking,:);
%         samples_updated=selected_samples(1:size(selected_samples,1)-size(new_points,1),:);
%     else
%         indices_to_remove=[];
%         selected_samples=train_fea_incremental;
%         selected_labels=train_fea_class_incremental;
%         samples_updated=selected_samples;
%     end
%     classes=unique(train_fea_class_incremental);
%     %     try
%     %     ix1=find(train_fea_class_incremental==classes(1));
%     %     catch
%     %         ix1=[];
%     %     end
%     %     try
%     %     ix2=find(train_fea_class_incremental==classes(2));
%     %     catch
%     %         ix2=[];
%     %     end
%     %determine how many samples to select from each class
%     nr_samples1=ceil(nr_samples/2);
%     nr_samples2=nr_samples-nr_samples1;
%     indices_to_remove=[];
%     [ranking,values,current_D,kernel,updated_sample,updated_class] = MAED_incremental(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,indices_to_remove,current_D,size(train_fea_incremental,1)+size(new_points,1),options,warping);
%
%     try
%         ix_up_class1=find(updated_class==classes(1));
%     catch
%         ix_up_class1=[];
%     end
%     try
%         ix_up_class2=find(updated_class==classes(2));
%     catch
%         ix_up_class2=[];
%     end
%     if nr_samples1>size(ix_up_class1,1)
%         nr_samples1=size(ix_up_class1,1);
%         nr_samples2=nr_samples-nr_samples1;
%     end
%
%     if nr_samples2>size(ix_up_class2,1)
%         nr_samples2=size(ix_up_class2,1);
%         nr_samples1=nr_samples-nr_samples2;
%     end
%
%     current_sample=[updated_sample(ix_up_class1(1:nr_samples1),:);updated_sample(ix_up_class2(1:nr_samples2),:)];
%     current_labels=[updated_class(ix_up_class1(1:nr_samples1),:);updated_class(ix_up_class2(1:nr_samples2),:)];
%     [ranking,values,current_D,kernel]=MAED(current_sample,current_labels,nr_samples,options,data_limit,warping);
%
%     %[ranking,values,current_D,kernel,current_sample,current_labels] = MAED_incremental_1(train_fea_incremental,train_fea_class_incremental,new_points,new_classes,current_D,nr_samples,options);
% end
% end
%
%
