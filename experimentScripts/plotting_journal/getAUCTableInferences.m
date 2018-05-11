function []= getAUCTableInferences(pathToResults,samplesN,batchSize,approaches,specificResult,outputPath)
aucs=containers.Map('KeyType','int32','ValueType','Any')
counter=1;
apprAuc={};
SRKDA=[];
SRDA=[];
SVM=[];
DT=[];
Ridge=[];

stdSRKDA=[];
stdSRDA=[];
stdSVM=[];
stdDT=[];
stdRidge=[];

for j=1:length(approaches)
    if strcmp(approaches{j},'lssvm')
        paths=sprintf('%s/smp_%d/bs_%d/%s/auc.mat',pathToResults,samplesN,batchSize,approaches{j});
    else
        paths=sprintf('%s/smp_%d/bs_%d/%s/%s/auc.mat',pathToResults,samplesN,batchSize,specificResult,approaches{j});
    end
    sprintf('PATH: %s',paths),exist(paths, 'file')
    if exist(paths, 'file')
        auc=load(paths)
        avgSRKDAAUC=auc.avgAucs;
        avgSRDAAUC=auc.SRDAAucs;
        avgDTAucs=auc.DTAucs;
        avgRidgeAucs=auc.RidgeAucs;
        avgSVMAucs=auc.SVMAucs;
        
        SRKDA(j)=nanmean(avgSRKDAAUC);
        SRDA(j)=max(nanmean(avgSRDAAUC),1-nanmean(avgSRDAAUC));
        SVM(j)=nanmean(avgSVMAucs);
        DT(j)=nanmean(avgDTAucs);
        Ridge(j)=nanmean(avgRidgeAucs);
        
        stdSRKDA(j)=nanstd(avgSRKDAAUC);
        stdSRDA(j)=max(nanstd(avgSRDAAUC),1-nanstd(avgSRDAAUC));
        stdSVM(j)=nanstd(avgSVMAucs);
        stdDT(j)=nanstd(avgDTAucs);
        stdRidge(j)=nanstd(avgRidgeAucs);

    else
        SRKDA(j)=NaN;
        SRDA(j)=NaN;
        SVM(j)=NaN;
        DT(j)=NaN;
        Ridge(j)=NaN;
        stdSRKDA(j)=NaN;
        stdSRDA(j)=NaN;
        stdSVM(j)=NaN;
        stdDT(j)=NaN;
        stdRidge(j)=NaN;
    end
    %apprAuc{j}=res;
end
latexHeader=sprintf('%s\n%s\n%s\n%s\n','\begin{table}[htp!]','\begin{center}','\begin{tabular}{r|llllll}','\multicolumn{1}{c}{Approach}& \multicolumn{1}{c|}{oMAED} & \multicolumn{1}{c}{bMAED} & \multicolumn{1}{c}{FSSVM} & \multicolumn{1}{c}{Random}\\')
latexBottom=sprintf('%s\n%s\n%s\n','\end{tabular}','\end{center}','\end{table}');

%oMAED
sprintf('\\multicolumn{1}{c}{SRKDA}&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f\\',SRKDA(1),stdSRKDA(1),SRKDA(2),stdSRKDA(2),SRKDA(3),stdSRKDA(3),SRKDA(4),stdSRKDA(4),SRKDA(5),stdSRKDA(5))
sprintf('\\multicolumn{1}{c}{SRDA}&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f\\',SRDA(1),stdSRDA(1),SRDA(2),stdSRDA(2),SRDA(3),stdSRDA(3),SRDA(4),stdSRDA(4),SRDA(5),stdSRDA(5))
sprintf('\\multicolumn{1}{c}{SVM}&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f\\',SVM(1),stdSVM(1),SVM(2),stdSVM(2),SVM(3),stdSVM(3),SVM(4),stdSVM(4),SVM(5),stdSVM(5))
sprintf('\\multicolumn{1}{c}{DT}&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f\\',DT(1),stdDT(1),DT(2),stdDT(2),DT(3),stdDT(3),DT(4),stdDT(4),DT(5),stdDT(5))
sprintf('\\multicolumn{1}{c}{Ridge}&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f&%0.2f$\\pm$%0.3f\\',Ridge(1),stdRidge(1),Ridge(2),stdRidge(2),Ridge(3),stdRidge(3),Ridge(4),stdRidge(4),Ridge(5),stdRidge(5))


% for j=1:length(approaches)
% approachAUC={};
% approachStdev={};
%     for i=1:length(samplesN)
%     results=aucs(samplesN(i));
%     apprRes=results{j};
%     approachAUC{i}=sprintf('%.0f',apprRes.auc*100);
%     approachStdev{i}=sprintf('%.1f',apprRes.stdev*100);
%     end
% resAUC(approaches{j})=approachAUC;
% resSTD(approaches{j})=approachStdev;
% end
% latexEntries={};
% resAUC.keys();
%  for j=1:length(approaches_order)
%      if isKey(resAUC,approaches_order{j})
%         aucs=resAUC(approaches_order{j});
%         stds=resSTD(approaches_order{j});
%         sprintf('\\multicolumn{1}{c}{%s}&%s$\\pm$%s&%s$\\pm$%s&%s$\\pm$%s&%s$\\pm$%s&%s$\\pm$%s&%s$\\pm$%s\\',approaches_order{j},char(aucs(1)),char(stds(1)),char(aucs(2)),char(stds(2)),char(aucs(3)),char(stds(3)),char(aucs(4)),char(stds(4)),char(aucs(5)),char(stds(5)))
%      end
%  end
end