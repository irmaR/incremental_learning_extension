function [results]=bSRDKA(settings,inferenceType)
start_tuning=tic;
[reguAlpha,reguBeta,kernelSigma]=tuneParams(settings,inferenceType);
tuningTime=toc(start_tuning)

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
s = RandStream('mt19937ar','Seed',settings.run);
fprintf('Running the learning...')
results=[];
results.tuningTime=tuningTime;
results.bestOptions=best_options;
results.reguAlpha=reguAlpha;
results.reguBeta=reguBeta;
results.sigma=kernelSigma;
[results]=MAEDBatch(settings.XTrain,settings.YTrain,settings.numSelectSamples,settings.batchSize,settings.dataLimit,options,settings.reportPoints,settings.balanced,inferenceType);
runtime=toc;
results.runtime=runtime;
end
