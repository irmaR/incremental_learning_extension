function []=plot_times_observations()

[y1i,y1b,y1l,e1i,e1b,e1l,x1]=get_times_observations('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/UCI_Adult/Balanced_Version1/smp_20/bs_100/Supervised/HeatKernel/k_0/','UCI Adult');
[y2i,y2b,y2l,e2i,e2b,e2l,x2]=get_times_observations('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/RCV/Balanced_Modified/Balanced_Modified/smp_20/bs_100/Supervised/HeatKernel/k_0/','RCV');
[y3i,y3b,y3l,e3i,e3b,e3l,x3]=get_times_observations('/Users/irma/Documents/MATLAB/RESULTS/Incremental_May/Incremental/USPS/Balanced_Modified/Balanced_Modified/smp_20/bs_100/Supervised/HeatKernel/k_0/','USPS')
 fig=figure(1)
subplot(1,3,1)
hold on
% errorbar(x1,y1i,e1i,'LineWidth',5,'Color','m')
% errorbar(x1,y1b,e1b,'LineWidth',5,'Color','b')
% errorbar(x1,y1l,e1l,'LineWidth',5,'Color','r')
plot(x1,y1i,'LineWidth',5,'Color','m')
plot(x1,y1b,'LineWidth',5,'Color','b')
plot(x1,y1l,'LineWidth',5,'Color','r')
set(gca,'xscale','log')
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('log #observed points','FontSize',35)
ylabel('Model Update Time (seconds)','FontSize',26)
xlim([x1(1) x1(length(x1))]);
title('UCI Adult')
%a = get(gca,'Children');
hold off

% 
subplot(1,3,2)
hold on
% errorbar(x3,y3i,e3i,'LineWidth',5,'Color','m')
% errorbar(x3,y3b,e3b,'LineWidth',5,'Color','b')
% errorbar(x3,y3l,e3l,'LineWidth',5,'Color','r')
plot(x3,y3i,'LineWidth',5,'Color','m')
plot(x3,y3b,'LineWidth',5,'Color','b')
plot(x3,y3l,'LineWidth',5,'Color','r')
set(gca,'xscale','log')
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlabel('log #observed points','FontSize',35)
title('USPS')
xlim([x3(1) x3(length(x3))]);
%b = get(gca,'Children');
hold off

subplot(1,3,3)
hold on
% errorbar(x2,y2i,e2i,'LineWidth',5,'Color','m')
% errorbar(x2,y2b,e2b,'LineWidth',5,'Color','b')
% errorbar(x2,y2l,e2l,'LineWidth',5,'Color','r')

plot(x2,y2i,'LineWidth',5,'Color','m')
plot(x2,y2b,'LineWidth',5,'Color','b')
plot(x2,y2l,'LineWidth',5,'Color','r')

set(gca,'xscale','log')
set(gca,'FontSize',20)
set(gca,'LooseInset',get(gca,'TightInset'))
xlim([0 x2(length(x2))]);
title('RCV')
xlabel('log #observed points','FontSize',35)
 b = get(gca,'Children');
hold off

h = [b]
lgd = legend(h,'F-LSSVM','SRKDA','i-SRKDA')
lgd.FontSize = 30;
lgd.Location = 'northwest';


end


function [mean_time_incr,mean_time_batch,mean_time_lssvm,stdev_time_inct,stdev_time_batch,stdev_time_lssvm,samples]=get_times_observations(path_to_results,title)
report_points=[];
times_incr=[];
times_batch=[];
samples=[];
path_to_incr=sprintf('%s/incr/results.mat',path_to_results);
path_to_batch=sprintf('%s/batch/results.mat',path_to_results);
path_to_lssvm=sprintf('%s/lssvm/results.mat',path_to_results);

path_to_incr1=sprintf('%s/incr/auc.mat',path_to_results);
path_to_batch1=sprintf('%s/batch/auc.mat',path_to_results);
path_to_lssvm1=sprintf('%s/lssvm/auc.mat',path_to_results);


counter=1;

if exist(path_to_incr, 'file') == 2
      aucs_inct=load(path_to_incr,'results');
      results=aucs_inct.results
      for i=1:length(results)
          if strcmp(title,'UCI Adult')
              times_incr(i,:)=results{i}.processing_times;
          else
              times_incr(i,:)=results{i}.processing_time;
          end
            if strcmp(title,'RCV') || strcmp(title,'USPS')
                samples=load(path_to_incr1,'report_points');
                samples=samples.report_points;
            else
                samples=results{i}.report_points;
            end
      end 
end

if exist(path_to_batch, 'file') == 2
    aucs_inct=load(path_to_batch,'results');
    
    results=aucs_inct.results;
    for i=1:length(results)
        if strcmp(title,'UCI Adult')
              times_batch(i,:)=results{i}.processing_times;
          else
              times_batch(i,:)=results{i}.processing_time;
        end  
    end
end

if exist(path_to_lssvm, 'file') == 2
    aucs_inct=load(path_to_lssvm,'results');
    results=aucs_inct.results;
    for i=1:length(results)
        if strcmp(title,'UCI Adult')
              times_lssvm(i,:)=results{i}.processing_times;
          else
              times_lssvm(i,:)=results{i}.processing_time;
          end
    end
end
times_lssvm
mean_time_incr=mean(times_incr);
mean_time_batch=mean(times_batch);
mean_time_lssvm=mean(times_lssvm);
stdev_time_inct=std(times_incr);
stdev_time_batch=std(times_batch);
stdev_time_lssvm=std(times_lssvm);
end


function [mean_time_incr,mean_time_batch,mean_time_lssvm,stdev_time_inct,stdev_time_batch,stdev_time_lssvm,samples]=get_processing_times_observations(path_to_results)
report_points=[];
times_incr=[];
times_batch=[];
samples=[];
path_to_incr=sprintf('%s/incr/results.mat',path_to_results);
path_to_batch=sprintf('%s/batch/results.mat',path_to_results);
path_to_lssvm=sprintf('%s/lssvm/results.mat',path_to_results);


counter=1;

if exist(path_to_incr, 'file') == 2
    aucs_inct=load(path_to_incr,'results');
    results=aucs_inct.results;
    for i=1:length(results)
       times_incr(i,:)=results{i}.selection_times;
       samples=results{i}.report_points;
    end
end

if exist(path_to_batch, 'file') == 2
    aucs_inct=load(path_to_batch,'results');
    results=aucs_inct.results;
    for i=1:length(results)
       try
       times_batch(i,:)=results{i}.selection_times;
       samples=results{i}.report_points;
       catch
       times_batch(i,:)=NaN;  
       end
       
    end
end

if exist(path_to_lssvm, 'file') == 2
    aucs_inct=load(path_to_lssvm,'results');
    results=aucs_inct.results;
    for i=1:length(results)
       times_lssvm(i,:)=results{i}.selection_times;
       samples=results{i}.report_points;
    end
end

mean_time_incr=mean(times_incr);
mean_time_batch=mean(times_batch);
try
mean_time_lssvm=mean(times_lssvm);
stdev_time_lssvm=std(times_lssvm);
catch
mean_time_lssvm=ones(1,size(mean_time_incr,1))*NaN;
stdev_time_lssvm=ones(1,size(mean_time_incr,1))*NaN;
end
stdev_time_inct=std(times_incr);
stdev_time_batch=std(times_batch);



end


