clear all
pathToData='/home/irma/work/DATA/incremental_learning/UCI/UCI_folds.mat'

load(pathToData)
%select first 3000 to be our data
r=1;
c=2;
nrSamples=500;
dataLimit=1000;
batchSize=500;



trainData=folds{r}.train;
trainClass=folds{r}.train_class;
testData=folds{r}.test;
testClass=folds{r}.test_class;
s = RandStream('mt19937ar','Seed',1);
ix=randperm(s,size(trainData,1))';
trainData=trainData(ix(1:dataLimit),:);
trainClass=trainClass(ix(1:dataLimit),:);
%standardize the training and test data
[trainData,min_train,max_train]=standardizeX(trainData);

testData=standardize(testData,min_train,max_train);
trainClass(trainClass~=c)=-1;
trainClass(trainClass==c)=1;
testClass(testClass~=c)=-1;
testClass(testClass==c)=1;
trainData=trainData(1:dataLimit,:);
trainClass=trainClass(1:dataLimit,:);


options = [];
options.KernelType = 'Gaussian';
options.t = 0.5;
options.bLDA=1;
options.ReguBeta=0.1;
options.ReguAlpha = 0.01;
options.k=0;
options.WeightMode='HeatKernel';
options.NeighborMode='Supervised';
%options.test=settings.XTest;
%options.test_class=settings.YTest;
%model.X=[];
%model.Y=[];
%model=batchUpdateModelBalanced(model,options,trainData,trainClass,nrSamples)
%auc=srkdaInference(model.K,model.X,model.Y,testData,testClass,options);
%area=max(auc,1-auc);

reportPoints=[nrSamples:batchSize:dataLimit]
settings.XTest=testData;
settings.YTest=testClass;
settings.XTrain=trainData;
settings.YTrain=trainClass;
settings.reguAlphaParams=[0.01];
settings.reguBetaParams=[0.1];
settings.kernelParams=[0.5];
settings.numSelectSamples=nrSamples;
settings.batchSize=batchSize;
settings.reportPoints=reportPoints;
settings.dataLimit=dataLimit;
settings.run=r;
settings.warping=1;
settings.balanced=1;
settings.weightMode=['HeatKernel'];
settings.neighbourMode=['Supervised'];
settings.ks=0;
settings.reportPointIndex=1;
res1=runExperiment(settings,'iSRKDA')
fprintf('Incremental finished')
res2=runExperiment(settings,'SRKDA')
