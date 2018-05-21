function [gamma,kernel_type,kernel]=tuneParamsLSSVM(settings)
kernel_type='RBF_kernel';
if length(settings.reguGammas)==1 && length(settings.kernelParams)==1
    gamma=settings.reguGammas(1);
    kernel=settings.kernelParams(1);
    kernel_type=kernel_type;
else
for i=1:length(settings.reguGammas)
    for j=1:length(settings.kernelParams)
        settings.gamma=settings.reguGammas(i);
        settings.kernel=settings.kernelParams(j);
        settings.kernel_type=kernel_type;
        %split training data into 5 folds for tuning the parameters
        folds=split_into_k_folds(settings.XTrain,settings.YTrain,5);
        performances=[];
        %go through each fold
        for k=1:length(folds)
            %increase batch size and interval for optimization
            increment=5;
            if settings.batchSize*increment>=settings.numSelectSamples
                batch_size_up=settings.numSelectSamples/2;
            else
                batch_size_up=settings.batchSize*increment;
            end
            interval_up=settings.batchSize*2;
            
            train_batch=folds{k}.train;
            train_batch_class=folds{k}.train_class;
            report_points_up=[settings.numSelectSamples:interval_up:size(folds{k}.train,1)-interval_up];
            
            %shuffle the data, splitting into folds might have messed up
            %the things and sorted the data
            s = RandStream('mt19937ar','Seed',settings.run);
            ix=randperm(s,size(train_batch,1))';
            train_batch=train_batch(ix,:);
            train_batch_class=train_batch_class(ix,:);
            options.test=folds{k}.test;
            options.test_class=folds{k}.test_class;
            oldReportPoints=settings.reportPoints;
            settings.reportPoints=report_points_up;
            [res]=FSLSSVM(settings,@lssvmInference);
            settings.reportPoints=oldReportPoints;
            aucs=[];
            for s=1:size(res.selectedDataPoints,1)
                area=lssvmInference(cell2mat(res.selectedDataPoints(s)),cell2mat(res.selectedLabels(s)),settings.XTest,settings.YTest,settings);
                fprintf('Area %f\t\n',area)
                aucs(s)=area;
            end
            performances(k)=mean(aucs);
            %end
        end
        area=mean(performances);
        validation_res(i,j)=area;
    end
end
%Get best options
[minp,ic] = max(validation_res,[],1);
[minminp,is] = max(minp);
[minmink,is1] = max(minminp);
ic=ic(is);
is=is(:,:,is1);
ic=ic(:,:,is1);
kernel = settings.kernelParams(ic);
gamma = settings.reguGammas(is);
end
end
