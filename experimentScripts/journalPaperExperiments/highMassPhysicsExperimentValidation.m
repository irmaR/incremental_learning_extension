function [parameters]=highMassPhysicsExperimentValidation(method,methodType,shuffleSeed,validationOffset,inferenceType,fidTrain,fidIndicesTrain,pathToResults,pathToCode,numSelectSamples,batchSize,dataLimit,warping,balanced,betas,alphas,kernels)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
NeighborModes='Supervised';
WeightModes='HeatKernel'
ks=0;
addpath(genpath(pathToCode))
reguBetaParams=betas;
reguAlphaParams=alphas;
kernelParams=kernels;

general_output=sprintf('%s/%s/validation/',pathToResults,method);
output_path=sprintf('%s/%s/validation/',pathToResults,method);
fprintf('Making folder %s',output_path)
mkdir(output_path)
param_info=sprintf('%s/params.txt',output_path)
fileID = fopen(param_info,'w');
fprintf(fileID,'Beta params=: ');
for i=1:length(reguBetaParams)
    fprintf(fileID,'%1.3f',reguBetaParams(i));
end
fprintf(fileID,'\n');
fprintf(fileID,'Alpha params: ');
for i=1:length(reguAlphaParams)
    fprintf(fileID,'%1.3f',reguAlphaParams(i));
end
fprintf(fileID,'\n');
fprintf(fileID,'Kernel params: ');
for i=1:length(kernelParams)
    fprintf(fileID,'%1.3f',kernelParams(i));
end
fprintf(fileID,'\n')
fprintf(fileID,'nr_samples:%d \n',numSelectSamples);
fprintf(fileID,'batch_size:%d \n',batchSize);
fprintf(fileID,'data_limit:%d \n',dataLimit);
fprintf(fileID,'Using warping?:%d \n',warping);
fprintf(fileID,'Using balancing?:%d \n',balanced);
fprintf('Batch sizE: %d',batchSize)
nrTrain=validationOffset;
batchReport=1000;

%get Train offset indices
fidTrain=fopen(fidTrain);
fidIndicesTrain=load(fidIndicesTrain);
fidIndicesTrain=fidIndicesTrain.arrayofOffsets;


%shuffle the array
s = RandStream('mt19937ar','Seed',shuffleSeed);
ix=randperm(s,size(fidIndicesTrain,1))';
fidIndicesTrain=fidIndicesTrain(ix(1:nrTrain,:),:);

reportPoints=[numSelectSamples:batchReport:nrTrain,nrTrain]

settings.initSample=[];
settings.initClass=[];
settings.reportPointIndex=1;
settings.indicesOffsetTrain=fidIndicesTrain;
settings.XTrainFileID=fidTrain;
settings.XTrain=fidTrain;
settings.formattingString='%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f';
settings.delimiter=',';
settings.reguAlphaParams=reguAlphaParams;
settings.reguBetaParams=reguBetaParams;
settings.kernelParams=kernelParams;
settings.numSelectSamples=numSelectSamples;
settings.batchSize=batchSize;
settings.reportPoints=reportPoints;
settings.dataLimit=dataLimit;
settings.run=1;
settings.warping=warping;
settings.balanced=balanced;
settings.weightMode=WeightModes;
settings.neighbourMode=NeighborModes;
settings.ks=ks;
settings.outputPath=output_path;

[settings.XTrain,settings.YTrain]=getDataInstancesSequential(fidTrain,settings.formattingString,settings.delimiter,settings.indicesOffsetTrain);
fprintf('%d Points for validation',size(settings.XTrain,1))
[reguAlpha,reguBeta,kernelSigma]=tuneParams(settings,methodType,inferenceType);


parameters.reguAlpha=reguAlphaParams;
parameters.reguBeta=reguBeta;
parameters.kernelSigma=kernelSigma;

fprintf('Saving results')
save(sprintf('%s/parameters.mat',output_path),'parameters');
fclose(fidTrain);
end