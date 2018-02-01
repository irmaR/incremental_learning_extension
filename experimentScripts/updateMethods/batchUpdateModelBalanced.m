function [model]=batchUpdateModelBalanced(model,options,XObserved,YObserved,numSamples)
model.X=XObserved;
model.Y=YObserved;
classes=unique(model.Y);
%determine how many samples to select from each class
nr_samples1=ceil(numSamples/2);
nr_samples2=numSamples-nr_samples1;
ix_up_class1=find(model.Y==classes(1));
ix_up_class2=find(model.Y==classes(2));
model1.X=model.X(ix_up_class1,:);
model2.X=model.X(ix_up_class2,:);
model1.Y=model.Y(ix_up_class1,:);
model2.Y=model.Y(ix_up_class2,:);

if nr_samples1>size(ix_up_class1,1)
    nr_samples1=size(ix_up_class1,1);
    nr_samples2=numSamples-nr_samples1;
end
if nr_samples2>size(ix_up_class2,1)
    nr_samples2=size(ix_up_class2,1);
    nr_samples1=numSamples-nr_samples2;
end
[model1,~] = MAED(model1,nr_samples1,options);
[model2,~] = MAED(model2,nr_samples2,options);
currentSample=[model1.X;model2.X];
currentLabels=[model1.Y;model2.Y];
model.X=currentSample;
model.Y=currentLabels;
[model,~]=MAED(model,numSamples,options);
