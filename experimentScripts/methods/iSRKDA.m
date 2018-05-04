function [results]=iSRKDA(settings,inferenceType)
start_tuning=tic;
[reguAlpha,reguBeta,kernelSigma]=tuneParams(settings,@MAEDIncremental,inferenceType);
tuningTime=toc(start_tuning)

if ~isfield(settings,'reportPointIndex')
    settings.reportPointIndex=1;
end
if ~isfield(settings,'initSample')
    settings.initSample=[];
end
if ~isfield(settings,'initClass')
    settings.initClass=[];
end
%Incrementally learn the model
options = [];
options.KernelType = 'Gaussian';
options.kernel_type = 'RBF_kernel';
options.kernel = kernelSigma;
options.gamma=1;
options.t = kernelSigma;
options.bLDA=settings.balanced;
options.ReguBeta=reguBeta;
options.ReguAlpha = reguAlpha;
options.k=settings.ks;
options.WeightMode=settings.weightMode;
options.NeighborMode=settings.neighbourMode;
options.test=settings.XTest;
options.test_class=settings.YTest;
options.positiveClass=settings.positiveClass;
options.classes=settings.classes;
%measure time
tic;
%shuffle data
s = RandStream('mt19937ar','Seed',settings.run);
ix=randperm(s,size(settings.XTrain,1))';
fprintf('Running the learning...')
sprintf('Run %d, Alpha: %f, Sigma: %f',settings.run,options.ReguAlpha,options.t)
fprintf('Init sample size %d-%d',size(settings.initSample,1),size(settings.initSample,2))
fprintf('Test sample size %d-%d',size(settings.XTest,1),size(settings.XTest,2))
best_options=options;
[results]=MAEDIncremental(settings,options,inferenceType);
results.tuningTime=tuningTime;
results.bestOptions=best_options;
results.reguAlpha=reguAlpha;
results.reguBeta=reguBeta;
results.sigma=kernelSigma;
runtime=toc;
results.runtime=runtime;
end