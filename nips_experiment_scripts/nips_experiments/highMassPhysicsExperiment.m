function []=highMassPhysicsExperiment(method,fidTrain,fidTest,fidIndicesTrain,fidIndicesTest,pathToResults,pathToCode,numSelectSamples,batchSize,dataLimit,warping,balanced,betas,alphas,kernels)
%USPS mat contains train,train_class,test and test_class
%we use one vs all strategy
NeighborModes='Supervised';
WeightModes='HeatKernel'
ks=0;

addpath(genpath(pathToCode))

reguBetaParams=betas;
reguAlphaParams=alphas;
kernelParams=kernels;

general_output=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/',pathToResults,numSelectSamples,batchSize);
output_path=sprintf('%s/smp_%d/bs_%d/%s/%s/k_%d/%s/',pathToResults,numSelectSamples,batchSize,method);
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

%nrTrain=7000000;
nrTrain=10000;
batchReport=1000;

settings.initSample=[];
settings.initClass=[];
reportPoints=[numSelectSamples:batchReport:nrTrain,nrTrain]
settings.reportPointIndex=1;

%get Train offset indices
fidTrain=fopen(fidTrain);
fidIndicesTrain=load(fidIndicesTrain)
fidIndicesTrain=fidIndicesTrain.arrayofOffsets;

%get Test offset indices
fidTest=fopen(fidTest);
fidIndicesTest=load(fidIndicesTest);
fidIndicesTest=fidIndicesTest.arrayofOffsets;

%sample a small subset from test data
ix=randperm(size(fidIndicesTest,1));
shuffledTest=fidIndicesTest(ix',:);

settings.indicesOffsetTrain=fidIndicesTrain;
settings.XTrainFileID=fidTrain;
settings.formattingString='%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f';
settings.delimiter=',';

[testData,testClass]=getDataInstancesSequential(fidTest,settings.formattingString,settings.delimiter,shuffledTest(1:2000));
%but train data is loaded sequentially and we only have indices for
%that

settings.XTest=testData;
settings.YTest=testClass;
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

results=runExperimentSequential(settings,method);
%save intermediate results just in case
fprintf('Saving results')
save(sprintf('%s/results.mat',output_path),'results');
end