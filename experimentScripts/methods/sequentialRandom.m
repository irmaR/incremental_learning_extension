function [results]=sequentialRandom(settings,inferenceType)
results=[];
start_tuning=tic;
settings1=settings;
settings1.indicesOffsetTrain=settings.indicesOffsetValidation;
settings1.batchSize=100;
settings1.read_size_test=100;
settings1.reportPoints=[settings1.numSelectSamples:200:size(settings1.indicesOffsetTrain,1)];
[settings1.XTrain,settings1.YTrain]=getDataInstancesSequential(settings1.XTrainFileID,settings1.formattingString,settings1.delimiter,settings1.indicesOffsetValidation);
fprintf('Validation size %d\n',size(settings1.XTrain,1))
fprintf('Train size settings %d\n',size(settings.indicesOffsetTrain,1))
fprintf('Train size settings1 %d\n',size(settings1.indicesOffsetTrain,1))
[reguAlpha,reguBeta,kernelSigma]=tuneParams(settings1,@MAEDIncrementalSequential,inferenceType);
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
%options.test=settings.XTest;
%options.test_class=settings.YTest;
best_options=options;
%measure time
tic;
results=[];
results.tuningTime=tuningTime;
results.bestOptions=best_options;
results.reguAlpha=reguAlpha;
results.reguBeta=reguBeta;
results.sigma=kernelSigma;
fprintf('Running the learning...')
sprintf('Run %d, Alpha: %f, Sigma: %f',settings.run,options.ReguAlpha,options.t)

[results]=randomSelectionSequential(settings,options,inferenceType);

runtime=toc;

results.runtime=runtime;
end