function []=plot_all_random_incremental_comparisons()
samples1=[20,40,60,80,100]
[random1,incr12,incr13,incr14]=plot_function_samples_1('/home/irma/work/RESULTS/Incremental_Journal_Experiments_May/UCI/',100,20,samples1,'iSRKDA');
[random3,incr32,incr33,incr34]=plot_function_samples_1('/home/irma/work/RESULTS/Incremental_Journal_Experiments_May/RCV/',100,40,samples1,'iSRKDA');
[random2,incr22,incr23,incr24]=plot_function_samples_1('/home/irma/work/RESULTS/Incremental_Journal_Experiments_May/USPS/',100,60,samples1,'iSRKDA')
fig=figure(1)

subplot(1,3,1)
hold on

plot(samples1,random1,'g+-','LineWidth',5);hold on;
plot(samples1,incr12,'kd','LineWidth',5)
plot(samples1,incr13,'cs','LineWidth',5)
plot(samples1,incr14,'mo','LineWidth',5)
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',25)
ylabel('AUC','FontSize',25)
ylim([0.5 1])
xlim([samples1(1) samples1(length(samples1))]);
title('UCI Adult')

a = get(gca,'Children');
hold off


subplot(1,3,2)
hold on

plot(samples1,random2,'g+-','LineWidth',5);hold on;
plot(samples1,incr22,'kd','LineWidth',5)
plot(samples1,incr23,'cs','LineWidth',5)
plot(samples1,incr24,'mo','LineWidth',5)
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('#selected samples','FontSize',25)
title('USPS')
xlim([samples1(1) samples1(length(samples1))]);
ylim([0.5 1])
b = get(gca,'Children');
hold off

% subplot(1,3,3)
% hold on
% plot(samples1,random3,'g+-','LineWidth',5);hold on;
% plot(samples1,incr32,'kd','LineWidth',5)
% plot(samples1,incr33,'cs','LineWidth',5)
% plot(samples1,incr34,'mo','LineWidth',5)
% set(gca,'FontSize',25)
% set(gca,'LooseInset',get(gca,'TightInset'))
% xlim([samples1(1) samples1(length(samples1))]);
% ylim([0.5 1])
% title('RCV')
% xlabel('#selected samples','FontSize',25)
% b = get(gca,'Children');
% hold off;

h=[b];
lgd = legend(h,'o-MAED @T3','o-MAED @T2','o-MAED @T1','random')
lgd.FontSize = 30;
lgd.Location = 'southeast';

saveas(fig,sprintf('/Users/irma/Documents/MATLAB/runtime.jpg'))
end

function [random_points,incr_bal_point1,incr_bal_point3,incr_bal_point4]=plot_function_samples_1(path_to_results,bs,division,samples,incremental)
for i=1:length(samples)
    %path_to_incr=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/incr_bal/auc.mat',path_to_results,samples(i),bs);
    path_to_incr=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/%s/auc.mat',path_to_results,samples(i),bs,incremental)
    path_to_random=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/random/auc.mat',path_to_results,samples(i),bs)
    
    if exist(path_to_incr, 'file') == 2
        %avg_auc_inct=load(path_to_incr,'avgAucs').avgAucs;
        %std_auc_inct=load(path_to_incr,'stdev');
        SRKDAAucs=load(path_to_incr)
        SRKDAAucs=load(path_to_incr,'SVMAucs')
        stdevSRKDAAucs=load(path_to_incr,'stdevSRKDAAucs');
        avg_aucs_inct{i}=SRKDAAucs.SVMAucs;
        std_aucs_inct{i}=stdevSRKDAAucs.stdevSRKDAAucs;
        report_points=load(path_to_incr,'reportPoints')
        nr_report_points=length(report_points.reportPoints)
    else
        avg_aucs_inct{i}=NaN;
        std_aucs_inct{i}=NaN;
    end
    
    if exist(path_to_random, 'file') == 2
        %avg_auc_rnd=load(path_to_random,'avgAucs').avgAucs;
        %std_auc_rnd=load(path_to_random,'stdev');
        avg_auc_rnd=load(path_to_random,'SVMAucs');
        std_auc_rnd=load(path_to_random,'stdevSRKDAAucs');
        avg_aucs_rnd{i}=avg_auc_rnd.SVMAucs
        std_aucs_rnd{i}=std_auc_rnd.stdevSRKDAAucs;
        report_points=load(path_to_incr,'reportPoints');
        nr_report_points=length(report_points.reportPoints)
    else
        avg_aucs_rnd{i}=NaN;
        std_aucs_rnd{i}=NaN;
    end
end
avg_aucs_inct
avg_aucs_rnd
random_points=[];
incr_bal_point1=[];
incr_bal_point2=[];
incr_bal_point3=[]
incr_bal_point4=[];
for i=1:length(samples)
    tmp=avg_aucs_rnd{i};
    random_points(i)=nanmean(tmp);
end
jump=floor(nr_report_points/division)
for i=1:length(samples)
    tmp=avg_aucs_inct{i}
    if isnan(tmp)
        incr_bal_point1(i)=NaN;
        incr_bal_point2(i)=NaN;
        incr_bal_point3(i)=NaN;
        incr_bal_point4(i)=NaN;
    else
        
        incr_bal_point1(i)=nanmean(tmp(2:2));
        incr_bal_point2(i)=nanmean(tmp(0+3*jump));
        incr_bal_point3(i)=nanmean(tmp(0+4*jump));
        incr_bal_point4(i)=nanmean(tmp(nr_report_points));
    end
end

end