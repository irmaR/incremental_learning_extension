function [reguAlpha,reguBeta,kernelSigma]=tuneParams(settings,methodType,inferenceType)
fprintf('Validation size %d tuning\n',size(settings.XTrain,1))
if length(settings.reguAlphaParams)==1 && length(settings.kernelParams)==1 && length(settings.reguBetaParams)==1
    reguAlpha = settings.reguAlphaParams(1);
    kernelSigma = settings.kernelParams(1);
    reguBeta = settings.reguBetaParams(1);
else
    for i=1:length(settings.reguAlphaParams)
        for j=1:length(settings.kernelParams)
            for b=1:length(settings.reguBetaParams)
                settings.ReguBeta=settings.reguBetaParams(b);
                settings.ReguType = 'Ridge';
                settings.ReguAlpha = settings.reguAlphaParams(i);
                settings.kernel=settings.kernelParams(j);
                settings.t=settings.kernelParams(j);
                fprintf('Params: Beta: %d, Alpha: %d, Kernel: %d\n',settings.ReguBeta,settings.ReguAlpha,settings.kernel)
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
                    settings.XTest=folds{k}.test;
                    settings.YTest=folds{k}.test_class;
                    [res]=methodType(settings,inferenceType);
                    res
                    aucs=[];
                    for s=1:size(res.selectedKernels,2)
                        area=srdaInference(cell2mat(res.selectedKernels(1,s)),cell2mat(res.selectedDataPoints(1,s)),cell2mat(res.selectedLabels(1,s)),folds{k}.test,folds{k}.test_class,settings);
                        fprintf('Area %f\t\n',area)
                        aucs(s)=area;
                    end
                    performances(k)=mean(aucs);
                end
                area=mean(performances);
                validation_res(i,j,b)=area;
            end
        end
    end
    %Get best options
    [minp,ic] = max(validation_res,[],1);
    [minminp,is] = max(minp);
    [minmink,is1] = max(minminp);
    ic=ic(is);
    is=is(:,:,is1);
    ic=ic(:,:,is1);
    reguAlpha = settings.reguAlphaParams(ic);
    kernelSigma = settings.kernelParams(is);
    reguBeta = settings.reguBetaParams(is1);
end

end