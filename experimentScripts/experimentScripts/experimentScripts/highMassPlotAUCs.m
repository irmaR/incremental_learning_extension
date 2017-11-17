function []=highMassPlotAUCs(results,runs)
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
    for run=1:runs
        resultsPath=sprintf('%s/%s/run%d/results.mat',results,methods{i},run)
        exist(resultsPath, 'file')
        if ~(exist(resultsPath, 'file') == 2)
            fprintf('Doesnt exist')
            continue
        end
        res=load(resultsPath);
        res=res.results;
        if size(reportPoints,2)==0
            reportPoints=res.reportPoints;
            nrReportPoints=size(cell2mat(res.selectedAUCs),2);
        end
        bestAUCs(run,:)=cell2mat(res.selectedAUCs);
        trueAUCs(run,:)=cell2mat(res.AUCs);
        avgBestAUCs(methods{i})=mean(bestAUCs);
        avgTrueAUCs(methods{i})=mean(trueAUCs);
        stdBestAUCs(methods{i})=std(bestAUCs);
        stdTrueAUCs(methods{i})=std(trueAUCs);
    end
    
end

counter=size(keys(avgBestAUCs),2);
methods=keys(avgBestAUCs)
colors('iSRKDA')={'blue'};
colors('SRKDA')={'red'};
colors('random')={'green'};
figure;
hold on;
reportPoints=reportPoints(1:nrReportPoints);
xlabel('# Observed data points')
ylabel('Best AUC')
for i=1:counter
    avgBestAUCs(methods{i})
    stdBestAUCs(methods{i})
    methods{i}
    C=colors(methods{i})
    
    errorbar(reportPoints,avgBestAUCs(methods{i}),stdBestAUCs(methods{i}),'LineWidth',2.5,'Color',C{1})
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
    C=colors(methods{i});
    errorbar(reportPoints,avgTrueAUCs(methods{i}),stdTrueAUCs(methods{i}),'LineWidth',2.5,'Color',C{1})
    legendInfo{i} = [methods{i}];
    legend(legendInfo,'FontSize',25,'Location', 'Best');
    ylim([0 1]);
end
hold off;

