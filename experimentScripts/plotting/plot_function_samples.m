function []=plot_function_samples(path_to_results,title_text)
samples=[10,20,40,60,80,100];
bs=100;


for i=1:length(samples)
    %path_to_incr=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/incr_bal/auc.mat',path_to_results,samples(i),bs);
    path_to_incr=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/incr_bal/auc.mat',path_to_results,samples(i),bs);

    path_to_random=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/rnd/auc.mat',path_to_results,samples(i),bs);
    report_points=load(path_to_incr,'report_points')
    nr_report_points=length(report_points.report_points)
    
    if exist(path_to_incr, 'file') == 2
    avg_auc_inct=load(path_to_incr,'avg_aucs');
    std_auc_inct=load(path_to_incr,'stdev');
    avg_aucs_inct{i}=avg_auc_inct.avg_aucs;
    std_aucs_inct{i}=std_auc_inct.stdev;  
    
    end

    if exist(path_to_random, 'file') == 2
    avg_auc_rnd=load(path_to_random,'avg_aucs');
    std_auc_rnd=load(path_to_random,'stdev');
    avg_aucs_rnd{i}=avg_auc_rnd.avg_aucs;
    std_aucs_rnd{i}=std_auc_rnd.stdev;
    report_points=load(path_to_incr,'report_points');          
    end
end
random_points=[];
incr_bal_point1=[];
incr_bal_point2=[];
incr_bal_point3=[]
incr_bal_point4=[];
for i=1:length(samples)
    tmp=avg_aucs_rnd{i};
    random_points(i)=nanmean(tmp);
end
jump=floor(nr_report_points/60)
for i=1:length(samples)
    tmp=avg_aucs_inct{i}
    incr_bal_point1(i)=nanmean(tmp(2:2));
    incr_bal_point2(i)=nanmean(tmp(0+3*jump));
    incr_bal_point3(i)=nanmean(tmp(0+4*jump));
    incr_bal_point4(i)=nanmean(tmp(nr_report_points));
end

fig=figure
plot(samples,random_points,'g+-','LineWidth',5);hold on;
plot(samples,incr_bal_point1,'kd','LineWidth',5);hold on;
plot(samples,incr_bal_point3,'cs','LineWidth',5);hold on;
plot(samples,incr_bal_point4,'mo','LineWidth',5);hold on;
legendInfo{1} = ['random'];
legendInfo{2} = ['i-SRKDA'];
xlabel('#selected samples','FontSize',20)
ylabel('AUC','FontSize',20)
legend(legendInfo);
ylim([0.5 1]);
%set(gca,'yscale','log')
set(gca,'FontSize',20)

b = get(gca,'Children');
h=[b];
lgd = legend(h,'incr @T3','incr @T2','incr @T1','random')
lgd.FontSize = 30;
lgd.Location = 'southeast';


end