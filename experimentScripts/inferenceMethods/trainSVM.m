function [auc]=trainSVM(Xtrain,Ytrain,Xtest,Ytest,categories,positiveClass,options)
opts = statset('MaxIter',30000);
% Train the classifier
svmStruct = svmtrain(Xtrain,Ytrain,'kernel_function','rbf','kktviolationlevel',0,'options',opts,'rbf_sigma',options.t);
% Make a prediction for the test set
Y_svm = svmclassify(svmStruct,Xtest);
[xx,yy,~,auc] = perfcurve(Ytest, Y_svm,positiveClass);
end