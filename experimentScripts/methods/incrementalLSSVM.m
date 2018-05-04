function [results]=incrementalLSSVM(settings)
start_tuning=tic;
[gamma,kernel_type,kernel]=tuneParamsLSSVM(settings);
tuningTime=toc(start_tuning)
fprintf('Running the learning...')
options = [];
options.KernelType = 'Gaussian';
options.kernel_type = kernel_type;
options.kernel = kernel;
options.gamma=gamma;
options.test=settings.XTest;
options.test_class=settings.YTest;
options.positiveClass=settings.positiveClass;
options.classes=settings.classes;
tic;
[results]=FSLSSVM(settings,options,@lssvmInference);
results.tuningTime=tuningTime;
results.bestOptions=options;
runtime=toc;
results.runtime=runtime;
end