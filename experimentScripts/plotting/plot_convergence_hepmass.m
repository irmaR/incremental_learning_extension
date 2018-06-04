pathiSRKDA1='/home/irma/work/RESULTS/Incremental_Journal_Experiments_May/smp_100/bs_1000/iSRKDA/results.mat'
pathRandom='/home/irma/work/RESULTS/Incremental_Journal_Experiments_May/smp_100/bs_1000/random/results.mat'

isrkda=load(pathiSRKDA1);
random=load(pathRandom);
reportPoints1=isrkda.results.reportPoints;

aucsiSRKDA=cell2mat(isrkda.results.AUCSVM);
aucRandom=cell2mat(random.results.AUCSVM);
plot(reportPoints1,aucsiSRKDA,'m','LineWidth',5);hold on;
plot(reportPoints1,aucRandom,'g','LineWidth',5)
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#observations','FontSize',25)
ylabel('AUC','FontSize',30)
ylim([0 1])
xlim([reportPoints1(1) reportPoints1(length(reportPoints1))]);
reportPoints1(length(reportPoints1))
title('HEPMASS')
a = get(gca,'Children');
hold off


h=[a];
lgd = legend(h,'random','o-MAED')
lgd.FontSize = 30;
lgd.Location = 'best';
