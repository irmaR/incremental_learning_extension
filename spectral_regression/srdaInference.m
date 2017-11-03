function [area]=srdaInference(kernel,trainFea,trainLabels,testFea,testLabels,options)
   %if we only have one class, return area=0
   if length(unique(trainLabels))==1
       fprintf('only one label selected!\n')
       unique(selected_tr_labels)
       fprintf('\n')
       area=NaN;
       return
   end
   options.ReguType = 'Ridge';
   [eigvector, elapseKSR] = SR(options, trainLabels,trainFea);
   Yhat = testFea*eigvector;
   if sum(isnan(Yhat))~=0
       area=0;
   else
    size(testFea)
    size(eigvector)
    size(Yhat)
   [X,Y,T,area] = perfcurve(testLabels,Yhat,'1');
   end
end