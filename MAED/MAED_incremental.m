function [experiment_info,current_sample,kernel]=MAED_incremental(data,labels,numSample,batch_size,options)
% MAED_incremental: Incremental Manifold Adaptive Experimental Design
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
%                         encountered during the incremental learnin, a
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
%   ReguBeta    -  regularization paramter for manifold
%                                 adaptive kernel.
%
%   ReguAlpha   -  ridge regularization paramter. Default 0.01
%Output:
%
%        experiment info (if specified throug observation points flag in options)     - The index of the sample which should be labeled.
%        current sample     - Final data points retained by the incremental
%                             approach
%        current_kernel     - Final kernel model


experiment_info={};
starting_count=tic;
%shuffle data
ix=randperm(size(data,1));
[current_sample,current_labels,current_D,kernel]=initialize_sample(options,data(ix,:),labels(ix,:),numSample);

point=1;

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

%main incremental loop
for j=0:batch_size:(size(data,1)-numSample-batch_size)
    starting_count1=tic;
    new_points=data(numSample+j+1:numSample+j+batch_size,:);
    new_classes=labels(numSample+j+1:numSample+j+batch_size,:);
    old_sample=current_sample;
    old_labels=current_labels;
    old_kernel=kernel;
    
    indices_to_remove=[];
    [current_D,kernel,current_sample,current_labels] = MAED_rank_incremental(current_sample,current_labels,new_points,new_classes,indices_to_remove,current_D,numSample,options);
    %if specified, keep the current model only if it improves the performance from the
    %previous model
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

    function [sample,labels,D,ranking,kernel]=initialize_sample(options,data,labels,model_size)
        %initial model: select first model_size points from the data
        train_fea_incremental=data(1:model_size,:);
        train_fea_class_incremental=labels(1:model_size,:);
        [ranking,~,D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,size(train_fea_incremental,1),options);
        sample=train_fea_incremental;
        labels=train_fea_class_incremental;
    end

    function [Dist,K,updated_sample,updated_class] = MAED_rank_incremental(original_sample,original_sample_class,new_data_point,new_data_point_class,indices_to_remove,D,selectNum,options)
        nSmp = size(original_sample,1);
        splitLabel = false(nSmp,1);
        if isfield(options,'splitLabel')
            splitLabel = options.splitLabel;
        end
        %I changed here
        %fprintf('Current size of D is %d',size(D,1))
        if isempty(indices_to_remove)
            Dist = EuDist2([original_sample;new_data_point],[],0);
            updated_sample=[original_sample;new_data_point];
            updated_class=[original_sample_class;new_data_point_class];
            nSmp=size(updated_sample,1);
        elseif isempty(new_data_point)
            Dist = EuDist2(original_sample,[],0);
            updated_sample=original_sample;
            updated_class=original_sample_class;
            nSmp=size(updated_sample,1);
        else
            [Dist,updated_sample,updated_class]=EuDist2_incremental(original_sample,original_sample_class,D,indices_to_remove,new_data_point,new_data_point_class,0);
            nSmp=size(updated_sample,1);
        end
        K = constructKernel_incremental(Dist,options);
        if isfield(options,'ReguBeta') && options.ReguBeta > 0
            
            if isfield(options,'W')
                W = options.W;
            else
                if isfield(options,'k')
                    Woptions.k = options.k;
                else
                    Woptions.k = 0;
                end
                
                Woptions.bLDA=options.bLDA;
                Woptions.t = options.t;
                Woptions.NeighborMode = options.NeighborMode ;
                Woptions.gnd = updated_class ;
                Woptions.WeightMode = options.WeightMode  ;
                W = constructW(updated_sample,Woptions);
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
                if isempty(indices_to_remove)
                    updated_sample=updated_sample(sampleList,:);
                    updated_class=updated_class(sampleList,:);
                    Dist=Dist(sampleList,sampleList);
                    K=K(sampleList,sampleList);
                end
                
            otherwise
                error('Optimization method does not exist!');
        end
    end

end

