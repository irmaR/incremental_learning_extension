data=load('data/data_adult.mat')
options={};
options.kernel_type='RBP_kernel';
options.t=5;
options.k=5;

model_size=50;
batch_size=5;
data_limit=500;

%running batch version
[batch_sample,batch_labels,batch_model]=MAED_batch(adult_fea,adult_class,model_size,batch_size,options,data_limit);

%running balanced batch version
[batch_bal_sample,batch_bal_labels,batch_bal_model]=MAED_batch_balanced(adult_fea,adult_class,model_size,batch_size,options,data_limit);

%running incremental version
[incr_sample,incr_labels,incr_model]=MAED_incremental(adult_fea,adult_class,model_size,batch_size,options,data_limit);

%running balanced incremental version
[incr_bal_sample,incr_bal_labels,incr_bal_model]=MAED_incremental_balanced(adult_fea,adult_class,model_size,batch_size,options,data_limit);