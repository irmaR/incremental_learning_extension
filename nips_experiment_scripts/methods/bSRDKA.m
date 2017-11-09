function [results]=bSRDKA(settings,inferenceType)
results=[];
validation_res=zeros(length(settings.reguAlphaParams),length(settings.kernelParams),length(settings.reguBetaParams));
k=1;
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
%shuffle data
s = RandStream('mt19937ar','Seed',settings.run);
ix=randperm(s,size(settings.XTrain,1))';
training_data=settings.XTrain(ix,:);
training_class=settings.YTrain(ix,:);
fprintf('Running the learning...')
[res]=MAEDBatch(settings.XTrain,settings.YTrain,settings.numSelectSamples,settings.batchSize,settings.dataLimit,options,settings.reportPoints,settings.balanced,inferenceType);

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
