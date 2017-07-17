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
    [current_sample,current_labels,kernel]=update_model_balanced(train_fea_incremental,train_fea_class_incremental,numSample,options);
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

    function [current_sample,current_labels,kernel]=update_model_balanced(train_fea_incremental,train_fea_class_incremental,numSample,options)
        %we assume that it's always binary problem, hence we split the data into
        %two classes
        classes=unique(train_fea_class_incremental);        
        %determine how many samples to select from each class
        nr_samples1=ceil(numSample/2);
        nr_samples2=numSample-nr_samples1;
        
        ix_up_class1=find(train_fea_class_incremental==classes(1));
        ix_up_class2=find(train_fea_class_incremental==classes(2));
        
        data_sub1=train_fea_incremental(ix_up_class1,:);
        data_sub2=train_fea_incremental(ix_up_class2,:);
        labels_sub1=train_fea_class_incremental(ix_up_class1,:);
        labels_sub2=train_fea_class_incremental(ix_up_class2,:);
        [r1,~] = MAED_batch_ranking(data_sub1,labels_sub1,nr_samples1,options);        
        [r2,~] = MAED_batch_ranking(data_sub2,labels_sub2,nr_samples2,options);
        current_sample=[data_sub1(r1,:);data_sub2(r2,:)];
        current_labels=[labels_sub1(r1,:);labels_sub2(r2,:)];
        [~,kernel] = MAED_batch_ranking(current_sample,current_labels,numSample,options);
        
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

