function [model]=batchUpdateModel(model,options,XNew,YNew,numSamples,dataLimit)
model.X=[model.X;XNew];
model.Y=[model.Y;YNew];
if size(model.X,1)>=dataLimit
    ix=randperm(size(model.X,1));
    model.X=model.X(ix(1:options.dataLimit),:);
    model.Y=model.Y(ix(1:options.dataLimit),:);
end
[model,values]=MAED(model,numSamples,options);
end
