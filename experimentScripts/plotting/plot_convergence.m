pathiSRKDA='/home/irma/work/RESULTS/Incremental/UCI/smp_100/bs_100/Supervised/HeatKernel/k_0/iSRKDA/auc.mat'
pathSRKDA='/home/irma/work/RESULTS/Incremental/UCI/smp_100/bs_100/Supervised/HeatKernel/k_0/SRKDA/auc.mat'

isrkda=load(pathiSRKDA);
srkda=load(pathSRKDA);

reportPoints=isrkda.reportPoints;
%aucsiSRKDA=isrkda.realAvgAUCs
%aucssrkda=srkda.realAvgAUCs;
aucsiSRKDA=isrkda.avgAucs
aucssrkda=srkda.avgAucs;

hold on
plot(reportPoints,aucsiSRKDA,'r','LineWidth',2)
plot(reportPoints,aucssrkda,'b','LineWidth',2)
%ylim([0 1])
y1=get(gca,'ylim')
plot([3000 3000],y1)
hold off;