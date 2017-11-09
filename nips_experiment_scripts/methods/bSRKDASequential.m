function [results]=bSRDKA(settings,inferenceType)
results=[];
validation_res=zeros(length(settings.reguAlphaParams),length(settings.kernelParams),length(settings.reguBetaParams));
k=1;
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
fprintf('Running the learning...')
[res]=MAEDBatchSequential(settings.XTrainFileID,settings.indicesOffsetTrain,settings.formattingString,settings.delimiter,settings.numSelectSamples,settings.batchSize,settings.reportPoints,settings.balanced,options,settings.dataLimit,inferenceType);

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
results.selectedBetas=res.selectedBetas;
results.realBetas=res.realBetas;
results.aucsReal=cell2mat(res.AUCs);
results.trainAUCs=cell2mat(res.trainAUCs);
results.tuningTime=tuningTime;
results.reportPoints=settings.reportPoints;
results.testPoints=settings.XTest;
results.testLabels=settings.YTest;
results.runtime=runtime;
fprintf('RESULTS')
end
