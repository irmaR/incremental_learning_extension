function [avg_runtimes_inct,std_runtimes_inct,avg_runtimes_batch,std_runtimes_batch,avg_runtimes_lssvm,std_runtimes_lssvm]=plot_runtimes_over_samples(path_to_results,title_text)
samples=[20,40,60];
bs=100;
avg_runtimes_inct=[];
std_runtimes_inct=[];

avg_runtimes_batch=[];
std_runtimes_batch=[];

avg_runtimes_lssvm=[];
std_runtimes_lssvm=[];

for i=1:length(samples)
    path_to_incr=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/iSRKDA/auc.mat',path_to_results,samples(i),bs);
    path_to_batch=sprintf('%s/smp_%d/bs_%d/Supervised/HeatKernel/k_0/SRKDA/auc.mat',path_to_results,samples(i),bs);
    path_to_lssvm=sprintf('%s/smp_%d/bs_%d/lssvm/auc.mat',path_to_results,samples(i),bs);
    if exist(path_to_incr, 'file') == 2
    avg_runtime_inct=load(path_to_incr,'avgRuntime')
    std_runtime_inct=load(path_to_incr,'stdRuntime')
    avg_runtimes_inct(i)=avg_runtime_inct.avgRuntime(end);
    std_runtimes_inct(i)=std_runtime_inct.stdRuntime(end);    
    else
       avg_runtimes_inct(i)=NaN;
       std_runtimes_inct(i)=NaN;
    end

    if exist(path_to_batch, 'file') == 2
    avg_runtime_batch=load(path_to_batch,'avgRuntime')
    std_runtime_batch=load(path_to_batch,'stdRuntime')
    avg_runtimes_batch(i)=avg_runtime_batch.avgRuntime(end);
    std_runtimes_batch(i)=std_runtime_batch.stdRuntime(end);
    else
       avg_runtimes_batch(i)=NaN;
       std_runtimes_batch(i)=NaN;
    end
    
    if exist(path_to_lssvm, 'file') == 2
    avg_runtime_lssvm=load(path_to_lssvm,'avgRuntime')
    std_runtime_lssvm=load(path_to_lssvm,'stdRuntime')
    avg_runtimes_lssvm(i)=avg_runtime_lssvm.avgRuntime(end);
    std_runtimes_lssvm(i)=std_runtime_lssvm.stdRuntime(end);
    else
       avg_runtimes_lssvm(i)=NaN;
       std_runtimes_lssvm(i)=NaN;
    end
        
end
end