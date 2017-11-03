function [results]=bSRDKA(settings,inferenceType)
results=[];
validation_res=zeros(length(settings.reguAlphaParams),length(settings.kernelParams),length(settings.reguBetaParams));
k=1;
start_tuning=tic;
if length(settings.reguAlphaParams)==1 && length(settings.kernelParams)==1 && length(settings.reguBetaParams)==1
    reguAlpha = settings.reguAlphaParams(1);
    kernelSigma = settings.kernelParams(1);
    reguBeta = settings.reguBetaParams(1);
    tuningTime=0;
else
    for i=1:length(settings.reguAlphaParams)
        for j=1:length(settings.kernelParams)
            for b=1:length(settings.reguBetaParams)
                options = [];
                options.KernelType = 'Gaussian';
                options.t = kernelParams(j);
                options.bLDA=blda;
                options.ReguType = 'Ridge';
                options.ReguBeta=reguBetaParams(b);
                options.ReguAlpha = reguAlphaParams(i);
                options.k=kNN;
                options.WeightMode=WeightMode;
                options.NeighborMode=NeighborMode;
                options.test=XTest;
                options.test_class=YTest;
                sprintf('Run %d, Alpha: %f, Sigma: %f',settings.run,options.ReguAlpha,options.t)
                %split training data into 5 folds for tuning the parameters
                folds=split_into_k_folds(settings.XTrain,settings.YTrain,5);
                performances=[];
                %go through each fold
                for k=1:length(folds)
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
                    report_points_up=[numSelectSamples:interval_up:size(folds{k}.train,1)-interval_up];
                    
                    %shuffle the data, splitting into folds might have messed up
                    %the things and sorted the data
                    s = RandStream('mt19937ar','Seed',run);
                    ix=randperm(s,size(train_batch,1))';
                    train_batch=train_batch(ix,:);
                    train_batch_class=train_batch_class(ix,:);
                    [res]=MAEDBatch(train_batch,train_batch_class,nr_samples,batch_size,options,report_points_up,data_limit,experiment_name,warping,inferenceType);
                    aucs=[];
                    for s=1:size(res.selectedKernels,1)
                        area=inferenceType(cell2mat(res.selectedKernels(s)),cell2mat(res.selectedDataPoints(s)),cell2mat(res.selectedLabels(s)),folds{k}.test,folds{k}.test_class,options);
                        fprintf('Area %f\t',area)
                        aucs(s)=area;
                    end
                    performances(k)=mean(aucs);
                    %end
                end
                area=mean(performances);
                validation_res(i,j,b)=area;
            end
        end
    end
    tuningTime=toc(start_tuning)
    fprintf('Performances')
    %Get best options
    [minp,ic] = max(validation_res,[],1);
    [minminp,is] = max(minp);
    [minmink,is1] = max(minminp);
    ic=ic(is);
    is=is(:,:,is1);
    ic=ic(:,:,is1);
    reguAlpha = reguAlphaParams(ic);
    kernelsigma = kernel_params(is);
    regu_beta = reguBetaParams(is1);
end
options = [];
options.KernelType = 'Gaussian';
options.t = kernelSigma;
options.bLDA=settings.balanced;
options.ReguBeta=reguBeta;
options.ReguAlpha = reguAlpha;
options.k=settings.ks;
options.WeightMode=settings.weightMode;
options.NeighborMode=settings.neighbourMode;
options.test=settings.XTest;
options.test_class=settings.YTest;
sprintf('Run %d, Alpha: %f, Sigma: %f',settings.run,options.ReguAlpha,options.t)
%measure time
tic;
%shuffle data
s = RandStream('mt19937ar','Seed',settings.run);
ix=randperm(s,size(settings.XTrain,1))';
training_data=settings.XTrain(ix,:);
training_class=settings.YTrain(ix,:);
fprintf('Running the learning...')
[res]=MAEDBatch(settings.XTrain,settings.YTrain,settings.numSelectSamples,settings.batchSize,settings.dataLimit,options,settings.reportPoints,settings.balanced,inferenceType);

runtime=toc;
best_options=options;
results.selectedPoints=res.selectedDataPoints;
results.selectedLabels=res.selectedLabels;
results.kernels=res.selectedKernels;
results.bestOptions=best_options;
results.validation_res=validation_res;
results.reguAlpha=reguAlpha;
results.processingTimes=res.processingTimes;
results.selectionTimes=res.times;
results.reguBeta=reguBeta;
results.sigma=kernelSigma;
results.aucs=cell2mat(res.selectedAUCs);
results.tuningTime=tuningTime;
results.reportPoints=settings.reportPoints;
results.testPoints=settings.XTest;
results.testLabels=settings.YTest;
results.runtime=runtime;
fprintf('RESULTS')
end
