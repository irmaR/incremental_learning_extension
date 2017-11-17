function [model]=batchUpdateModelBalanced(model,options,XObserved,YObserved,numSamples)
model.X=XObserved;
model.Y=YObserved;
%if size(model.X,1)>=dataLimit
%    ix=randperm(size(model.X,1));
%    model.X=model.X(ix(1:options.dataLimit),:);
%    model.Y=model.Y(ix(1:options.dataLimit),:);
%end
%we assume that it's always binary problem, hence we split the data into
%two classes
classes=unique(model.Y);
%ix1=find(model.Y==classes(1));
%ix2=find(model.Y==classes(2));

%determine how many samples to select from each class
nr_samples1=ceil(numSamples/2);
nr_samples2=numSamples-nr_samples1;

[model,values] = MAED(model,numSamples,options);
%[ranking,values,current_D,kernel] = MAED(train_fea_incremental,train_fea_class_incremental,nr_samples,options,data_limit,warping);
ix_up_class1=find(model.Y==classes(1));
ix_up_class2=find(model.Y==classes(2));
if nr_samples1>size(ix_up_class1,1)
    nr_samples1=size(ix_up_class1,1);
    nr_samples2=numSamples-nr_samples1;
end
if nr_samples2>size(ix_up_class2,1)
    nr_samples2=size(ix_up_class2,1);
    nr_samples1=numSamples-nr_samples2;
end
currentSample=[model.X(ix_up_class1(1:nr_samples1),:);model.X(ix_up_class2(1:nr_samples2),:)];
currentLabels=[model.Y(ix_up_class1(1:nr_samples1),:);model.Y(ix_up_class2(1:nr_samples2),:)];
model.X=currentSample;
model.Y=currentLabels;
[model,values]=MAED(model,numSamples,options);
