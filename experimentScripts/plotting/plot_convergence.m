pathiSRKDA1='/home/irma/work/RESULTS/Incremental/UCI/smp_100/bs_100/Supervised/HeatKernel/k_0/iSRKDA/auc.mat'
pathSRKDA1='/home/irma/work/RESULTS/Incremental/UCI/smp_100/bs_100/Supervised/HeatKernel/k_0/SRKDA/auc.mat'

pathiSRKDA2='/home/irma/work/RESULTS/Incremental/USPS/smp_100/bs_100/Supervised/HeatKernel/k_0/iSRKDA/auc.mat'
pathSRKDA2='/home/irma/work/RESULTS/Incremental/USPS/smp_100/bs_100/Supervised/HeatKernel/k_0/SRKDA/auc.mat'

pathiSRKDA3='/home/irma/work/RESULTS/Incremental/RCV/smp_100/bs_100/Supervised/HeatKernel/k_0/iSRKDA/auc.mat'
pathSRKDA3='/home/irma/work/RESULTS/Incremental/RCV/smp_100/bs_100/Supervised/HeatKernel/k_0/SRKDA/auc.mat'

isrkda1=load(pathiSRKDA1);
srkda1=load(pathSRKDA1);
reportPoints1=isrkda1.reportPoints;
aucsiSRKDA1=isrkda1.realAvgAUCs
aucssrkda1=srkda1.realAvgAUCs;
%aucsiSRKDA=isrkda.avgAucs
%aucssrkda=srkda.avgAucs;

isrkda2=load(pathiSRKDA2);
srkda2=load(pathSRKDA2);
reportPoints2=isrkda2.reportPoints;
aucsiSRKDA2=isrkda2.realAvgAUCs
aucssrkda2=srkda2.realAvgAUCs;
%aucsiSRKDA=isrkda.avgAucs
%aucssrkda=srkda.avgAucs;

isrkda3=load(pathiSRKDA3);
srkda3=load(pathSRKDA3);
reportPoints3=isrkda3.reportPoints;
aucsiSRKDA3=isrkda3.realAvgAUCs
aucssrkda3=srkda3.realAvgAUCs;
%aucsiSRKDA=isrkda.avgAucs
%aucssrkda=srkda.avgAucs;


subplot(1,3,1)
hold on
plot(reportPoints1,aucsiSRKDA1,'m','LineWidth',5);hold on;
plot(reportPoints1,aucssrkda1,'b','LineWidth',5)
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#observations','FontSize',25)
ylabel('AUC','FontSize',30)
ylim([0.5 1])
xlim([reportPoints1(1) reportPoints1(length(reportPoints1))]);
y1=get(gca,'ylim')
reportPoints1(length(reportPoints1))
set(gca,'XTick',0:1000:4000);
plot([1000 1000],y1,'g--','LineWidth',3)
title('UCI Adult')
a = get(gca,'Children');
hold off


subplot(1,3,2)
hold on
plot(reportPoints2,aucsiSRKDA2,'m','LineWidth',5);hold on;
plot(reportPoints2,aucssrkda2,'b','LineWidth',5);
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#observations','FontSize',25)
title('USPS')
ylim([0.5 1]);
xlim([reportPoints2(1) reportPoints2(length(reportPoints2))]);
y1=get(gca,'ylim')

plot([1000 1000],y1,'g--','LineWidth',3)
b = get(gca,'Children');
hold off

subplot(1,3,3)
hold on
plot(reportPoints3,aucsiSRKDA3,'m','LineWidth',5);hold on;
plot(reportPoints3,aucssrkda3,'b','LineWidth',5)
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
ylim([0.5 1])
xlim([reportPoints3(1) reportPoints3(length(reportPoints3))]);
y1=get(gca,'ylim')
plot([1000 1000],y1,'g--','LineWidth',3)
title('RCV')
xlabel('#observations','FontSize',25)
b = get(gca,'Children');
hold off;

h=[b];
lgd = legend(h,'data limit','SRKDA','i-SRKDA')
lgd.FontSize = 30;
lgd.Location = 'southeast';



