function [model]=incrementalUpdateModelBalanced(model,options,XNew,YNew,numSamples)
classes=unique(model.Y);
%determine how many samples to select from each class
nr_samples1=ceil(numSamples/2);
nr_samples2=numSamples-nr_samples1;
[model] = MAEDRankIncremental(model,XNew,YNew,size(model.X,1)+size(XNew,1),options);

try
    ix_up_class1=find(model.Y==classes(1));
catch
    ix_up_class1=[];
end
try
    ix_up_class2=find(model.Y==classes(2));
catch
    ix_up_class2=[];
end
if nr_samples1>size(ix_up_class1,1)
    nr_samples1=size(ix_up_class1,1);
    nr_samples2=numSamples-nr_samples1;
end

if nr_samples2>size(ix_up_class2,1)
    nr_samples2=size(ix_up_class2,1);
    nr_samples1=numSamples-nr_samples2;
end
current_sample=[model.X(ix_up_class1(1:nr_samples1),:);model.X(ix_up_class2(1:nr_samples2),:)];
current_labels=[model.Y(ix_up_class1(1:nr_samples1),:);model.Y(ix_up_class2(1:nr_samples2),:)];
model.X=current_sample;
model.Y=current_labels;
model=MAED(model,numSamples,options);
end