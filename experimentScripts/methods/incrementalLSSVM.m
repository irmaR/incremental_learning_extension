function [results]=incrementalLSSVM(settings)
start_tuning=tic;
[gamma,kernel_type,kernelSigma]=tuneParamsLSSVM(settings);
tuningTime=toc(start_tuning)
fprintf('Running the learning...')
settings.kernel = kernelSigma;
settings.gamma=gamma;
settings.t=kernelSigma;
options = [];
options.kernel=kernelSigma;
options.reguGamma=gamma;


%options.KernelType = 'Gaussian';
%options.kernel_type = kernel_type;
%options.kernel = kernel;
%options.gamma=gamma;
%options.test=settings.XTest;
%options.test_class=settings.YTest;
%options.positiveClass=settings.positiveClass;
%options.classes=settings.classes;
tic;
[results]=FSLSSVM(settings,@lssvmInference);
results.tuningTime=tuningTime;
results.bestOptions=options;
runtime=toc;
results.runtime=runtime;
end