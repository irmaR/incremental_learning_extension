function [resAUC,resSTD]= getAUCTable(pathToResults,samplesN,batchSize,approaches,specificResult,outputPath)
aucs=containers.Map('KeyType','int32','ValueType','Any')
approaches_order={'SRKDA','iSRKDA','SRDA','iSRDA','lssvm'}
counter=1;
for i=1:length(samplesN)
  apprAuc={};
  samplesN(i)
  for j=1:length(approaches)
      approaches{j}
      strcmp(approaches{j},'lssvm')
      if strcmp(approaches{j},'lssvm')
           paths=sprintf('%s/smp_%d/bs_%d/%s/auc.mat',pathToResults,samplesN(i),batchSize,approaches{j})
      else
           paths=sprintf('%s/smp_%d/bs_%d/%s/%s/auc.mat',pathToResults,samplesN(i),batchSize,specificResult,approaches{j})
      end
      if exist(paths, 'file')
          auc=load(paths);
          avgAUCs=auc.avgAucs;
          stdev=auc.stdev;
          res.appr=approaches{j};
          res.auc=avgAUCs(end);
          res.stdev=stdev(end);
          res.sampleN=samplesN(i);
      else
          res.appr=approaches{j};
          res.auc=NaN;
          res.stdev=NaN;
          res.sampleN=samplesN(i);
      end
      apprAuc{j}=res;
  end
  aucs(samplesN(i))=apprAuc;  
end
latexHeader=sprintf('%s\n%s\n%s\n%s\n','\begin{table}[htp!]','\begin{center}','\begin{tabular}{r|llllll}','\multicolumn{1}{c}{Approach}& \multicolumn{1}{c|}{20} & \multicolumn{1}{c}{40} & \multicolumn{1}{c}{60} & \multicolumn{1}{c}{80} & \multicolumn{1}{c}{100}\\')
resAUC=containers.Map('KeyType','char','ValueType','Any')
resSTD=containers.Map('KeyType','char','ValueType','Any')
latexBottom=sprintf('%s\n%s\n%s\n','\end{tabular}','\end{center}','\end{table}')



for j=1:length(approaches)
approachAUC={};
approachStdev={};
    for i=1:length(samplesN)
    results=aucs(samplesN(i));
    apprRes=results{j};
    apprRes.auc
    approachAUC{i}=sprintf('%.0f',apprRes.auc*100);
    approachStdev{i}=sprintf('%.0f',apprRes.stdev*100);
    end
resAUC(approaches{j})=approachAUC;
resSTD(approaches{j})=approachStdev;
end
latexEntries={}
for j=1:length(approaches_order)
    if isKey(resAUC,approaches_order{j})
       aucs=resAUC(approaches_order{j})
       stds=resSTD(approaches_order{j})
       size(approaches_order{j})
       
       approaches_order{j}
       sprintf('\multicolumn{1}{c}{%s}&%d$\pm$%d&%d$\pm$%d&%d$\pm$%d&%d$\pm$%f*&%f$\pm$%f\\')
       latexEntries{j}=sprintf('\multicolumn{1}{c}{%s}&%d$\pm$%d&%d$\pm$%d&%d$\pm$%d&%d$\pm$%f*&%f$\pm$%f\\',approaches_order{j},aucs(1),stds(1),aucs(2),stds(2),aucs(3),std(3),aucs(4),stds(4),aucs(5),stds(5))
       

    end
end

end