function [output_path]=highMassPhysicsExperimenttraining(method,run,shuffleSeedValidation,validationOffset,fidTrain,fidTest,fidIndicesTrain,fidIndicesTest,pathToResults,pathToCode,numSelectSamples,batchSize,dataLimit,warping,balanced,betas,alphas,kernels)
NeighborModes='Supervised';
WeightModes='HeatKernel'
ks=0;
reguBetaParams=betas;
reguAlphaParams=alphas;
kernelParams=kernels;
output_path=sprintf('%s/%s/run%d/',pathToResults,method,run);
fprintf('Making folder %s',output_path);
mkdir(output_path);
param_info=sprintf('%s/params.txt',output_path);
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

%get Train offset indices
fidTrain=fopen(fidTrain);
fidIndicesTrain=load(fidIndicesTrain);
fidIndicesTrain=fidIndicesTrain.arrayofOffsets;
%shuffle the array according to the validation seed
sVal=RandStream('mt19937ar','Seed',shuffleSeedValidation);
ix=randperm(sVal,size(fidIndicesTrain,1));
%shuffle the array according to the run seed
%Load training data indices and shuffle
%Reserve one random part of train for selecting the model: this will be
%called validation
s = RandStream('mt19937ar','Seed',run);
ix=randperm(s,size(fidIndicesTrain,1));
fidIndicesTrain=fidIndicesTrain(ix',:);
fidIndicesValidation=fidIndicesTrain(ix(1:validationOffset)',:);
fidIndicesTrain=fidIndicesTrain(ix((validationOffset+1))',:);

nrTrain=size(fidIndicesTrain,1);

%Load testing data indices and shuffle
fidTest=fopen(fidTest);
fidIndicesTest=load(fidIndicesTest);
fidIndicesTest=fidIndicesTest.arrayofOffsets;
ix=randperm(s,size(fidIndicesTest,1));
shuffledTest=fidIndicesTest(ix',:);


batchReport=5000;
settings.initSample=[];
settings.initClass=[];
reportPoints=[numSelectSamples:batchReport:nrTrain];
settings.reportPointIndex=1;


settings.indicesOffsetTrain=fidIndicesTrain;
settings.XTrainFileID=fidTrain;
settings.formattingString='%s%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f';
settings.delimiter=',';

size(fidIndicesValidation)
settings.indicesOffsetValidation=fidIndicesValidation;

settings.indicesOffsetTest=fidIndicesTest;
settings.XTestFileID=fidTest;
settings.read_size_test=400;
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

results=runExperimentSequential(settings,method);
fprintf('Saving results')
save(sprintf('%s/results.mat',output_path),'results');
fclose(fidTrain);
fclose(fidTest);
end