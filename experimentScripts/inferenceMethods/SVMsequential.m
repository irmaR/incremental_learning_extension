function [auc]=SVMsequential(selected_tr_points,selected_tr_labels,settings,options)
batch=settings.read_size_test;
pointerObs=1;
opts = statset('MaxIter',30000);
% Train the classifier
svmStruct = svmtrain(selected_tr_points,selected_tr_labels,'kernel_function','rbf','kktviolationlevel',0,'options',opts,'rbf_sigma',options.t);

predictions=[];
test_labels=[];
while 1
    starting_count1=tic;
    if pointerObs+batch>=size(settings.indicesOffsetTest,1) 
        batch=size(settings.indicesOffsetTest,1)-pointerObs;
    end
   [XNew,YNew]=getDataInstancesSequential(settings.XTestFileID,settings.formattingString,settings.delimiter,settings.indicesOffsetTest(pointerObs:pointerObs+batch));
   if(size(XNew,1)==0)
       break
   end
   predictions=[predictions;svmclassify(svmStruct,XNew)];
   test_labels=[test_labels;YNew];
   pointerObs=pointerObs+batch+1;
end
[xx,yy,~,auc] = perfcurve(test_labels, predictions,settings.positiveClass);
end