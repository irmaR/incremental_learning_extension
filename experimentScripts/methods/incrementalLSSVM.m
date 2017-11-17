function [results]=incrementalLSSVM(settings)
start_tuning=tic;
[gamma,kernel_type,kernel]=tuneParamsLSSVM(settings);
tuningTime=toc(start_tuning)
fprintf('Running the learning...')
options = [];
options.kernel_type = kernel_type;
options.kernel = kernel;
options.gamma=gamma;
options.test=settings.XTest;
options.test_class=settings.YTest;

[results]=FSLSSVM(settings,options,@lssvmInference);
results.tuningTime=tuningTime;
results.bestOptions=options;
runtime=toc;
results.runtime=runtime;
end