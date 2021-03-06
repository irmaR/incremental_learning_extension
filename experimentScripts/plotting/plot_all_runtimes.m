clear all
close all
samples=[20,40,60,80,100];
[y1i,e1i,y1b,e1b,y1l,e1l]=plot_runtimes_over_samples(samples,'/home/irma/work/RESULTS/Incremental_Journal_Experiments_May/UCI/','UCI')
[y2i,e2i,y2b,e2b,y2l,e2l]=plot_runtimes_over_samples(samples,'/home/irma/work/RESULTS/Incremental_Journal_Experiments_May/RCV/','RCV');
[y3i,e3i,y3b,e3b,y3l,e3l]=plot_runtimes_over_samples(samples,'/home/irma/work/RESULTS/Incremental_Journal_Experiments_May/USPS/','USPS');

subplot(1,3,1)
hold on
errorbar(samples,y1i,e1i,'LineWidth',5,'Color','m');
errorbar(samples,y1b,e1b,'LineWidth',5,'Color','b');
errorbar(samples,y1l,e1l,'LineWidth',5,'Color','r');
set(gca,'yscale','log')
set(gca,'FontSize',25)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',35)
ylabel('log Runtime (seconds)','FontSize',30)
xlim([samples(1) samples(length(samples))]);
title('UCI Adult')
a = get(gca,'Children');
hold off


subplot(1,3,2)
hold on
errorbar(samples,y3i,e3i,'LineWidth',5,'Color','m')
errorbar(samples,y3b,e3b,'LineWidth',5,'Color','b')
errorbar(samples,y3l,e3l,'LineWidth',5,'Color','r')
set(gca,'yscale','log')
set(gca,'FontSize',25)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',35)
title('USPS')
xlim([samples(1) samples(length(samples))]);
b = get(gca,'Children');
hold off

subplot(1,3,3)
hold on
errorbar(samples,y2i,e2i,'LineWidth',5,'Color','m')
errorbar(samples,y2b,e2b,'LineWidth',5,'Color','b')
errorbar(samples,y2l,e2l,'LineWidth',5,'Color','r')

set(gca,'yscale','log')
set(gca,'FontSize',25)
set(gca,'LooseInset',get(gca,'TightInset'))
xlim([samples(1) samples(length(samples))]);
title('RCV')
xlabel('#selected samples','FontSize',35)
%b = get(gca,'Children');
hold off

y1i,e1i,y1b,e1b,y1l,e1l

h = [b]
lgd = legend(h,'F-LSSVM','b-MAED','o-MAED')
lgd.FontSize = 28;
lgd.Location = 'southeast';

%saveas(fig,sprintf('/Users/irma/Documents/MATLAB/runtime.jpg'))

%rect = [0.25, 0.25, .25, .25];
%set(h, 'Position', rect)