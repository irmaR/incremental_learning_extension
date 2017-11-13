fidTrain='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/train/all_train.csv';
fidTest='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/test/all_test.csv';
offsetTrain='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/offsetIndices.mat';
offsetTest='/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/offsetIndicesTest.mat';
output='/Users/irma/Documents/MATLAB/RESULTS/Test/HighMassNoMass/';
codePath='/Users/irma/Documents/MATLAB/CODE_local/incremental_learning_extension';

for run=1:3
     output='/Users/irma/Documents/MATLAB/RESULTS/Test/HighMassNoMass/';
     highMassPhysicsExperiment('SRKDA',run,fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,50,200,2000,1,1,[0.0001],[0.001],[5]);
     highMassPhysicsExperiment('iSRKDA',run,fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,50,200,2000,1,1,[0.0001],[0.001],[5]);
     highMassPhysicsExperiment('random',run,fidTrain,fidTest,offsetTrain,offsetTest,output,codePath,50,200,2000,1,1,[0.0001],[0.001],[5]);
end

methods={'SRKDA','iSRKDA','random'};
reportPoints=[];
nrReportPoints=0;


avgBestAUCs=containers.Map('KeyType','char','ValueType','Any');
avgTrueAUCs=containers.Map('KeyType','char','ValueType','Any');
stdBestAUCs=containers.Map('KeyType','char','ValueType','Any');
stdTrueAUCs=containers.Map('KeyType','char','ValueType','Any');


for i=1:size(methods,2)
    trueAUCs=[];
    bestAUCs=[];
    for run=1:3
        resultsPath=sprintf('/Users/irma/Documents/MATLAB/RESULTS/Test/HighMassNoMass/smp_50/bs_200/%s/run%d/results.mat',methods{i},run);
        res=load(resultsPath);
        res=res.results;
        if size(reportPoints,2)==0
            reportPoints=res.reportPoints;
            nrReportPoints=size(res.aucsReal,2);
        end
        bestAUCs(run,:)=res.aucs;
        trueAUCs(run,:)=res.aucsReal;
    end
    avgBestAUCs(methods{i})=mean(bestAUCs);
    avgTrueAUCs(methods{i})=mean(trueAUCs);
    stdBestAUCs(methods{i})=std(bestAUCs);
    stdTrueAUCs(methods{i})=std(trueAUCs);
end

counter=size(methods,2);
colorVec = hsv(counter);
figure;
hold on;
reportPoints=reportPoints(1:nrReportPoints);
xlabel('# Observed data points')
ylabel('Best AUC')
for i=1:counter
    %errorbar(report_points.report_points,results{1,i}.avg_aucs,stdevs{1,i}.stdev)
    %length(report_points.report_points)
    errorbar(reportPoints,avgBestAUCs(methods{i}),stdBestAUCs(methods{i}),'LineWidth',2.5,'Color',colorVec(i,:))
    legendInfo{i} = [methods{i}];
    legend(legendInfo,'FontSize',25,'Location', 'Best');
    ylim([0 1]);
end
hold off;

figure;
hold on;
reportPoints=reportPoints(1:nrReportPoints);
xlabel('# Observed data points')
ylabel('True AUC')
for i=1:counter
    %errorbar(report_points.report_points,results{1,i}.avg_aucs,stdevs{1,i}.stdev)
    %length(report_points.report_points)
    errorbar(reportPoints,avgTrueAUCs(methods{i}),stdTrueAUCs(methods{i}),'LineWidth',2.5,'Color',colorVec(i,:))
    legendInfo{i} = [methods{i}];
    legend(legendInfo,'FontSize',25,'Location', 'Best');
    ylim([0 1]);
end
hold off;

%plotAUCvsObservedPoints('/Users/irma/Documents/MATLAB/RESULTS/Test/HighMassNoMass/smp_50/bs_200/',{'iSRKDA','SRKDA','random'},'/Users/irma/Documents/MATLAB/RESULTS/Test/HighMassNoMass/smp_50/bs_200/AUCvsObservedPoints.png')