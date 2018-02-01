pathToResults='/Users/irma/Documents/MATLAB/RESULTS/Spirals/'
smpSize=[10,50];
batchSize=10;
datasetSizes=[5000,10000,20000,40000,80000];

runtimesiSRKDA=[];
runtimesSRKDA=[];

for i=1:size(datasetSizes,2)
    pathToResultiSRKDA=sprintf('%s/spiral%d/smp_%d/bs_%d/iSRKDA/results.mat',pathToResults,datasetSizes(i),smpSize,batchSize)
    pathToResultSRKDA=sprintf('%s/spiral%d/smp_%d/bs_%d/SRKDA/results.mat',pathToResults,datasetSizes(i),smpSize,batchSize)
    if exist(pathToResultiSRKDA, 'file') == 2
        iSRKDARes=load(pathToResultiSRKDA);
        runtimesiSRKDA(i)=iSRKDARes.res.runtime;
    end
    if exist(pathToResultSRKDA, 'file') == 2
        SRKDARes=load(pathToResultSRKDA);
        runtimesSRKDA(i)=SRKDARes.res.runtime;
    else
        runtimesSRKDA(i)=NaN;
    end
end
figure
plot(datasetSizes,runtimesiSRKDA,datasetSizes,runtimesSRKDA)
