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
best_options=options;
%shuffle data
fprintf('Running the learning...')
results=[];
results.tuningTime=tuningTime;
results.bestOptions=best_options;
results.reguAlpha=reguAlpha;
results.reguBeta=reguBeta;
results.sigma=kernelSigma;
[results]=MAEDBatchSequential(settings,options,inferenceType);
runtime=toc;
results.runtime=runtime;
fprintf('RESULTS')
end
