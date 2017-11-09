function [results]=iSRDKA(settings,inferenceType)
results=[];
start_tuning=tic;
[reguAlpha,reguBeta,kernelSigma]=tuneParams(settings,inferenceType);
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
options.t = kernelSigma;
options.bLDA=settings.balanced;
options.ReguBeta=reguBeta;
options.ReguAlpha = reguAlpha;
options.k=settings.ks;
options.WeightMode=settings.weightMode;
options.NeighborMode=settings.neighbourMode;
options.test=settings.XTest;
options.test_class=settings.YTest;
%measure time
tic;
%shuffle data
s = RandStream('mt19937ar','Seed',settings.run);
ix=randperm(s,size(settings.XTrain,1))';
fprintf('Running the learning...')
sprintf('Run %d, Alpha: %f, Sigma: %f',settings.run,options.ReguAlpha,options.t)
fprintf('Init sample size %d-%d',size(settings.initSample,1),size(settings.initSample,2))
fprintf('Train sample size %d-%d',size(settings.XTrain,1),size(settings.XTrain,2))
[res]=MAEDIncremental([settings.initSample;settings.XTrain(ix,:)],[settings.initClass;settings.YTrain(ix,:)],settings.numSelectSamples,settings.batchSize,options,settings.reportPoints,settings.reportPointIndex,settings.balanced,inferenceType);
runtime=toc;
best_options=options;
results.selectedPoints=res.selectedDataPoints;
results.selectedLabels=res.selectedLabels;
results.finalSample=res.selectedDataPoints{1,res.reportPointIndex};
results.finalSample
results.finalClass=res.selectedLabels{1,res.reportPointIndex};
results.finalKernel=res.selectedKernels{1,res.reportPointIndex};
results.kernels=res.selectedKernels;
results.bestOptions=best_options;
results.reguAlpha=reguAlpha;
results.processingTimes=res.processingTimes;
results.selectionTimes=res.times;
results.reguBeta=reguBeta;
results.sigma=kernelSigma;
results.selectedBetas=res.selectedBetas;
results.realBetas=res.realBetas;
results.aucs=cell2mat(res.selectedAUCs);
results.aucsReal=cell2mat(res.AUCs);
results.trainAUCs=cell2mat(res.trainAUCs);
results.tuningTime=tuningTime;
results.percentageRemoved=res.percentageRemoved;
results.reportPoints=settings.reportPoints;
results.testPoints=settings.XTest;
results.testLabels=settings.YTest;
results.runtime=runtime;
results.reportPointIndex=res.reportPointIndex;
end