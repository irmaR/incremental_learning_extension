function [resAUC,resSTD]= getAUCTable(pathToResults,samplesN,approaches,specificResult,outputPath)
aucs=containers.Map('KeyType','int32','ValueType','Any')
counter=1;
for i=1:length(samplesN)
  apprAuc={};
  for j=1:length(approaches)
      paths=sprintf('%s/smp_%d/%s/%s/auc.mat',pathToResults,samplesN(i),specificResult,approaches{j})
      auc=load(paths);
      avgAUCs=auc.avgAucs;
      stdev=auc.stdev;
      res.appr=approaches{j};
      res.auc=avgAUCs(end);
      res.stdev=stdev(end);
      apprAuc{j}=res;
  end
  aucs(samplesN(i))=apprAuc;  
end

latexHeader='\\begin{table}[htp!] \n \\begin{center} \n\\begin{tabular}{r|llllll} \n \\multicolumn{1}{c}{} & \\multicolumn{5}{c}{USPS}\\\\ \n \\multicolumn{1}{c}{Approach}& \\multicolumn{1}{c|}{20} & \\multicolumn{1}{c}{40} & \\multicolumn{1}{c}{60} & \\multicolumn{1}{c}{80} & \\multicolumn{1}{c}{100}\\\\'
resAUC=containers.Map('KeyType','char','ValueType','Any')
resSTD=containers.Map('KeyType','char','ValueType','Any')

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

end