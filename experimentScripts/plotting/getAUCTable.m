function [resAUC,resSTD]= getAUCTable(pathToResults,samplesN,batchSize,approaches,specificResult,outputPath)
aucs=containers.Map('KeyType','int32','ValueType','Any')
approaches_order={'SRKDA','iSRKDA','SRDA','iSRDA','lssvm','random'}
counter=1;
for i=1:length(samplesN)
  apprAuc={};
  samplesN(i)
  for j=1:length(approaches)
      if strcmp(approaches{j},'lssvm')
           paths=sprintf('%s/smp_%d/bs_%d/%s/auc.mat',pathToResults,samplesN(i),batchSize,approaches{j});
      else
           paths=sprintf('%s/smp_%d/bs_%d/%s/%s/auc.mat',pathToResults,samplesN(i),batchSize,specificResult,approaches{j});
      end
      sprintf('PATH: %s',paths),exist(paths, 'file')
      if exist(paths, 'file')
          auc=load(paths)
          avgAUCs=auc.realAvgAUCs
          stdev=auc.stdevReal;
          res.appr=approaches{j};
          res.auc=avgAUCs(end)
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
resAUC=containers.Map('KeyType','char','ValueType','Any');
resSTD=containers.Map('KeyType','char','ValueType','Any');
latexBottom=sprintf('%s\n%s\n%s\n','\end{tabular}','\end{center}','\end{table}');

for j=1:length(approaches)
approachAUC={};
approachStdev={};
    for i=1:length(samplesN)
    results=aucs(samplesN(i));
    apprRes=results{j};
    approachAUC{i}=sprintf('%.0f',apprRes.auc*100);
    approachStdev{i}=sprintf('%.1f',apprRes.stdev*100);
    end
resAUC(approaches{j})=approachAUC;
resSTD(approaches{j})=approachStdev;
end
latexEntries={};
resAUC.keys();
 for j=1:length(approaches_order)
     if isKey(resAUC,approaches_order{j})
        aucs=resAUC(approaches_order{j});
        stds=resSTD(approaches_order{j});
        sprintf('\\multicolumn{1}{c}{%s}&%s$\\pm$%s&%s$\\pm$%s&%s$\\pm$%s&%s$\\pm$%s&%s$\\pm$%s&%s$\\pm$%s\\',approaches_order{j},char(aucs(1)),char(stds(1)),char(aucs(2)),char(stds(2)),char(aucs(3)),char(stds(3)),char(aucs(4)),char(stds(4)),char(aucs(5)),char(stds(5)))
     end
 end
end