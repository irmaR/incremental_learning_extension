function [res]=main_run(experimentName,fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,numSelectSamples,batchSize,dataLimit,validationOffset,runs)
w = warning ('off','all');
resultsPath=highMass(experimentName,fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,runs,validationOffset,numSelectSamples,batchSize,dataLimit)
end