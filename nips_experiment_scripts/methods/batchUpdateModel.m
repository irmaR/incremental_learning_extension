function [model]=batchUpdateModel(model,options,XObserved,YObserved,numSamples)
model.X=XObserved;
model.Y=YObserved;
[model,values]=MAED(model,numSamples,options);
end
