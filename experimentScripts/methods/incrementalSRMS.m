function [results]=incrementalSRMS(settings)
start_tuning=tic;
[reguAlpha,reguBeta,kernelSigma]=tuneParams(settings,@MAEDIncremental,@srkdaInference);
tuningTime=toc(start_tuning)
if ~isfield(settings,'reportPointIndex')
    settings.reportPointIndex=1;
end
if ~isfield(settings,'initSample')
    settings.initSample=[];
end
if ~isfield(settings,'initClass')
    settings.initClass=[];
end
settings.kernel = kernelSigma;
settings.ReguBeta=reguBeta;
settings.ReguAlpha = reguAlpha;
settings.t=kernelSigma;
options.kernel=kernelSigma;
options.reguBeta=reguBeta;
options.reguAlpha=reguAlpha;
settings.alpha=5;
tic;
best_options=options;[results]=SRMSelection(settings,@srkdaInference);
results.tuningTime=tuningTime;
results.bestOptions=best_options;
results.reguAlpha=reguAlpha;
results.reguBeta=reguBeta;
results.sigma=kernelSigma;
runtime=toc;
results.runtime=runtime;
end