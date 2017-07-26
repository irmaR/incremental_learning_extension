function [experiment_info,model]=MAED_incremental(data,labels,num_samples,batch_size,options)
% MAED_incremental: Incremental Manifold Adaptive Experimental Design
%     sampleList = MAED(fea,selectNum,options)
% Input:
%   data               - Data matrix MXN, where M is the number of data
%                          points and N is then number of features
%   labels             - Labels for data (Mx1)
%   num_samples          - The size of the fixed-size model
%   options            - Struct value in Matlab. The fields in options
%                               that can be set:
%
%   observation_points -  an array consisting of desired observation
%                         snapshots of the algorithm. For
%                         example, model_observation_points=[50,100] will record
%                         results obtained after observing 50 and 100 points
%                         respectively.
%   plot_label_distr- flag and output path to a plot that collects label distribution during the online
%                         learning, and plot the trend at the end of the learning.
%                          if observation_points is set, the distribution will be recorded in the
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


experiment_info={};
starting_count=tic;
%shuffle data
ix=randperm(size(data,1));
data=data(ix,:);
labels=labels(ix,:);
[current_sample,current_labels,current_D,kernel]=initialize_sample(options,data,labels,batch_size,num_samples);

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
if isfield(options,'plot_label_distr')
    [a,b]=hist(current_labels,unique(current_labels));
    c=[b,a'];
    experiment_info.lists_of_label_distributions{point}=c;
end

%main incremental loop
for j=batch_size+1:batch_size:(size(data,1)-batch_size)
    fprintf('%d-%d,%d\n',j,j+batch_size-1,size(j:j+batch_size-1,2))
    starting_count1=tic;
    new_points=data(j:j+batch_size-1,:);
    new_classes=labels(j:j+batch_size-1,:);
    old_sample=current_sample;
    old_labels=current_labels;
    old_kernel=kernel;
    
    [current_D,kernel,current_sample,current_labels] = MAED_rank_incremental(current_sample,current_labels,new_points,new_classes,current_D,num_samples,options);
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
            experiment_info.list_of_kernels{point}=kernel;
            experiment_info.list_of_selected_times(point)=toc(starting_count);
            experiment_info.lists_of_processing_times(point)=toc(starting_count1);
            experiment_info.lists_of_areas{point}=current_area;
            if isfield(options,'plot_label_distr')
                [a,b]=hist(current_labels,unique(current_labels));
                c=[b,a'];
                experiment_info.lists_of_label_distributions{point}=c;
            end
        end
        if point<=length(model_observation_points) && options.model_size+j>=model_observation_points(point)
            point=point+1;
        end
    else
        if isfield(options,'plot_label_distr')
            [a,b]=hist(current_labels,unique(current_labels));
            c=[b,a'];
            experiment_info.lists_of_label_distributions{point}=c;
            point=point+1;
        end
    end
end

model.X=current_sample;
model.Y=current_labels;
model.K=kernel;

%plot label distribution if specified
if isfield(options,'plot_label_distr')
    plot_label_distributions(experiment_info.lists_of_label_distributions,options.plot_label_distr);
end

    function [sample,labels,D,ranking,kernel]=initialize_sample(options,data,labels,batch_size,model_size)
        %initial model: select first model_size points from the data
        train_fea_incremental=data(1:batch_size,:);
        train_fea_class_incremental=labels(1:batch_size,:);
        [ranking,~,D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,model_size,options);
        sample=train_fea_incremental(ranking,:);
        labels=train_fea_class_incremental(ranking,:);
    end

    function [Dist,K,updated_sample,updated_class] = MAED_rank_incremental(original_sample,original_sample_class,new_data_point,new_data_point_class,D,selectNum,options)
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
        nSmp = size(original_sample,1);
        splitLabel = false(nSmp,1);
        if isfield(options,'splitLabel')
            splitLabel = options.splitLabel;
        end

        [Dist,updated_sample,updated_class]=EuDist2_incremental(original_sample,original_sample_class,D,new_data_point,new_data_point_class,0);
        nSmp=size(updated_sample,1);
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


