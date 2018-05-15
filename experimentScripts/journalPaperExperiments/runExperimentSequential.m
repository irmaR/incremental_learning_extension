function [results]=runExperimentSequential(settings,method)
switch lower(method)
   case {lower('iSRDA')}
        results=iSRKDA(settings,@srdaInference);
    case {lower('iSRKDA')}
        results=iSRKDASequential(settings,@log_reg);    
    case {lower('SRDA')}
        results=bSRDKA(settings,@srdaInference);
    case {lower('SRKDA')}
        results=bSRKDASequential(settings,@srkdaInference);
    case {lower('batch')}
        results=incremental(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,data_limit,'batch',r,warping,blda,k,WeightMode,NeighborMode,@srdaInference);
    case {lower('random')}
        fprintf('Running random sampling balanced version')
        results=sequentialRandom(settings,@lssvmInference);
    case {lower('lssvm')}
        results=incremental_lssvm(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,nr_samples,interval,batch_size,report_points,data_limit,'lssvm',r);
end
end


function [results]=incremental_lssvm(training_data,training_class,test_data,test_class,kernel_params,gamma_params,nr_samples,interval,batch_size,report_points,data_limit,experiment_name,run)
results=[];
start_tuning=tic;
validation_res=zeros(length(kernel_params),length(gamma_params));
k=1;
kernel='RBF_kernel';
if length(kernel_params)==1 && length(kernel_params)==1
    gamma = gamma_params(1);
    kernel = kernel_params(1);
    tuning_time=0;
else
    start_tuning=tic;
    for i=1:length(kernel_params)
        for j=1:length(gamma_params)
            fprintf('HERE')
            options.gamma=gamma_params(j);
            options.kernel=kernel_params(i);
            options.kernel_type=kernel;
            options.test=test_data;
            options.test_class=test_class;
            folds=split_into_k_folds(training_data,training_class,5);
            performances=[];
            
            %go through each fold
            for k=1:length(folds)
                fprintf('Fold: %d',k)
                %increase batch size and interval for optimization
                increment=5;
                if batch_size*increment>=nr_samples
                    batch_size_up=nr_samples/2;
                else
                    batch_size_up=batch_size*increment;
                end
                interval_up=interval*2;
                
                train_batch=folds{k}.train;
                train_batch_class=folds{k}.train_class;
                report_points_up=[nr_samples:interval_up:size(folds{k}.train,1)-interval_up];
                
                %shuffle the data, splitting into folds might have messed up
                %the things and sorted the data
                s = RandStream('mt19937ar','Seed',run);
                ix=randperm(s,size(train_batch,1))';
                train_batch=train_batch(ix,:);
                train_batch_class=train_batch_class(ix,:);
                [selected_points,selected_labels,list_of_selected_times,lists_of_processing_times,~,~]=MAED_experiment_instance(train_batch,train_batch_class,nr_samples,batch_size,options,report_points_up,data_limit,experiment_name,0);
                aucs=[];
                fprintf('Points selected, running inference')
                for s=1:size(selected_points,1)
                    area=run_inference_lssvm(selected_points(s),folds{k}.train,folds{k}.train_class,selected_labels(s),folds{k}.test,folds{k}.test_class,options);
                    aucs(s)=area;
                end
                performances(k)=mean(aucs);
                %end
            end
            area=mean(performances);
            validation_res(i,j,b)=area;
        end
    end
    tuning_time=toc(start_tuning)
end

fprintf('Performances')
%Get best options
[minp,ic] = max(validation_res,[],1);
[minminp,is] = max(minp);
[minmink,is1] = max(minminp);
ic=ic(is);
is=is(:,:,is1);
ic=ic(:,:,is1);
kernel = kernel_params(ic);
gamma = gamma_params(is);
options = [];
options.kernel_type = 'RBF_kernel';
options.kernel = kernel;
options.gamma=gamma;
options.test=test_data;
options.test_class=test_class;
sprintf('Run %d, kernel: %f, gamma: %f',run,options.kernel,options.gamma)
%measure time
tic;
%shuffle data
s = RandStream('mt19937ar','Seed',run);
ix=randperm(s,size(training_data,1))';
training_data=training_data(ix,:);
training_class=training_class(ix,:);
[selected_points,selected_labels,list_of_selected_times,lists_of_processing_times,~,~,lists_of_areas]=MAED_experiment_instance(training_data,training_class,nr_samples,batch_size,options,report_points,data_limit,experiment_name,0);
runtime=toc
best_options=options;
results.selected_points=selected_points;
results.selected_labels=selected_labels;
results.best_options=best_options;
results.validation_res=validation_res;
results.kernel=kernel;
results.gamma=gamma;
results.processing_times=lists_of_processing_times;
results.selection_times=list_of_selected_times;
results.aucs=cell2mat(lists_of_areas);
results.tuning_time=tuning_time;
results.report_points=report_points;
results.test_points=test_data;
results.test_labels=test_class;
results.runtime=runtime;
fprintf('RESULTS')
end

function [area]=run_inference_lssvm(Xs,training_data,training_classes,Ys,test_data,test_class,options)
gamma=1;
%fprintf('sigma %f',options.t)
features = AFEm(Xs,options.kernel_type, options.kernel,Xs);
try,
    [CostL3, gamma_optimal] = bay_rr(features,Ys,options.gamma,1);
catch,
    warning('no Bayesian optimization of the regularization parameter');
    gamma_optimal = gamma;
end
[w,b] = ridgeregress(features,Ys,options.gamma);
Yh0 = AFEm(Xs,options.kernel_type, options.kernel,test_data)*w+b;
echo off;
[area,se,thresholds,oneMinusSpec,Sens]=roc(Yh0,test_class);
end



