function [auc]=trainRidge(Xtrain,Ytrain,Xtest,Ytest,categories,positiveClass)
[b, ~, ~, ~, ~, ~] = fastridge(full(Xtrain),full(Ytrain), 'lambda', 1);
y_ridge_matlab = full(Xtest)*b;
[xx,yy,~,auc] = perfcurve(Ytest, y_ridge_matlab,positiveClass);
end