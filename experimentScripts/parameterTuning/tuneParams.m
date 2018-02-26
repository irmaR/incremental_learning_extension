function [reguAlpha,reguBeta,kernelSigma]=tuneParams(settings,methodType,inferenceType)
if length(settings.reguAlphaParams)==1 && length(settings.kernelParams)==1 && length(settings.reguBetaParams)==1
    reguAlpha = settings.reguAlphaParams(1);
    kernelSigma = settings.kernelParams(1);
    reguBeta = settings.reguBetaParams(1);
else
    for i=1:length(settings.reguAlphaParams)
        for j=1:length(settings.kernelParams)
            for b=1:length(settings.reguBetaParams)
                options = [];
                options.KernelType = 'Gaussian';
                options.t = settings.kernelParams(j);
                options.bLDA=settings.balanced;
                options.ReguType = 'Ridge';
                options.ReguBeta=settings.reguBetaParams(b);
                options.ReguAlpha = settings.reguAlphaParams(i);
                options.k=settings.ks;
                options.WeightMode=settings.weightMode;
                options.NeighborMode=settings.neighbourMode;
                sprintf('Run %d, Alpha: %f, Sigma: %f',settings.run,options.ReguAlpha,options.t)
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
                    [res]=methodType(settings,options,inferenceType);                    
                    aucs=[];
                    for s=1:size(res.selectedKernels,1)
                        area=inferenceType(cell2mat(res.selectedKernels(s)),cell2mat(res.selectedDataPoints(s)),cell2mat(res.selectedLabels(s)),folds{k}.test,folds{k}.test_class,options);
                        fprintf('Area %f\t\n',area)
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