function [results]=incrementalSRMS(settings)
start_tuning=tic;
options = [];
options.KernelType = 'Gaussian';
options.kernel_type = 'RBF_kernel';
options.kernel = settings.kernelParams;
options.gamma=1;
options.test=settings.XTest;
options.test_class=settings.YTest;
options.positiveClass=settings.positiveClass;
options.classes=settings.classes;
options.alpha=5;
tic;
[results]=SRMSelection(settings,options,@srkdaInference);
results.tuningTime=tuningTime;
results.bestOptions=options;
runtime=toc;
results.runtime=runtime;
end