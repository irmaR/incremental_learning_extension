function []=plotAUCvsObservedPoints(pathToResults,methodNames,output)
BestAUCs=containers.Map('KeyType','char','ValueType','Any')
RealAUCs=containers.Map('KeyType','char','ValueType','Any')
reportPoints=[];
nrReportPoints=0;
for i=1:size(methodNames,2)
    pathToRes=sprintf('%s/%s/results.mat',pathToResults,methodNames{i});
    res=load(pathToRes);
    res=res.results;
    BestAUCs(methodNames{i})=res.aucs;
    RealAUCs(methodNames{i})=res.aucsReal;
    if size(reportPoints,2)==0
        reportPoints=res.reportPoints;
        nrReportPoints=size(res.aucsReal,2);
    end
end
counter=size(methodNames,2);
colorVec = hsv(counter);
figure;
hold on;
reportPoints=reportPoints(1:nrReportPoints);
xlabel('# Observed data points')
ylabel('Best AUC')
for i=1:counter
    %errorbar(report_points.report_points,results{1,i}.avg_aucs,stdevs{1,i}.stdev)
    %length(report_points.report_points)
    plot(reportPoints,BestAUCs(methodNames{i}),'LineWidth',2.5,'Color',colorVec(i,:))
    legendInfo{i} = [methodNames{i}];
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
    plot(reportPoints,RealAUCs(methodNames{i}),'LineWidth',2.5,'Color',colorVec(i,:))
    legendInfo{i} = [methodNames{i}];
    legend(legendInfo,'FontSize',25,'Location', 'Best');
    ylim([0 1]);
end
hold off;

end