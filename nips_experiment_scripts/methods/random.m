function [results]=random(settings,inferenceType)
results=[];
start_tuning=tic;
[reguAlpha,reguBeta,kernelSigma]=tuneParams(settings,inferenceType);
tuningTime=toc(start_tuning)

if ~isfield(settings,'XTrainFileID')
    error('Error. \n File handle for training data not specified. Cannot continue without that for the sequential run.')
end

if ~isfield(settings,'indicesOffsetTrain')
    error('Error. \n Offset indices for training data not specified. Cannot continue without that for the sequential run.')
end

if ~isfield(settings,'formattingString')
    error('Error. \n formatting string not known. We need that for the sequential')
end

if ~isfield(settings,'delimiter')
    error('Error. \n Delimiter string not known. We need that for the sequential')
end

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
fprintf('Running the learning...')
sprintf('Run %d, Alpha: %f, Sigma: %f',settings.run,options.ReguAlpha,options.t)
fprintf('Init sample size %d-%d',size(settings.initSample,1),size(settings.initSample,2))

[res]=randomSequential(settings.XTrainFileID,settings.indicesOffsetTrain,settings.formattingString,settings.delimiter,settings.numSelectSamples,settings.batchSize,settings.reportPoints,options,inferenceType);

runtime=toc;
best_options=options;
results.selectedPoints=res.selectedDataPoints;
results.selectedLabels=res.selectedLabels;
results.finalSample=res.selectedDataPoints{1,res.reportPointIndex};
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