function [auc]=trainDT(Xtrain,Ytrain,Xtest,Ytest,categories,positiveClass)

%# train classification decision tree
%Ytrain=Ytrain);Ytest=categorical(Ytest);
%categories
t = ClassificationTree.fit(full(Xtrain),Ytrain,'CategoricalPredictors',categories);
% Make a prediction for the test set
Y_t = t.predict(full(Xtest));
[xx,yy,~,auc] = perfcurve(Ytest, Y_t,positiveClass);
%figure;
%plot(xx,yy)
%xlabel('False positive rate');
%ylabel('True positive rate')
%title('ROC curve for ''yes'', predicted vs. actual response (Test Set)')
%text(0.5,0.25,{'TreeBagger with full feature set',strcat('Area Under Curve = ',num2str(auc))},'EdgeColor','k');
end