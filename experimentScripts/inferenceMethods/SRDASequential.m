function [auc]=SRDASequential(selected_tr_points,selected_tr_labels,settings,options,EvalSetIndices,XFileID)
batch=settings.read_size_test;
pointerObs=1;
options.ReguType = 'Ridge';
options.gnd = selected_tr_labels;
[eigvector, ~] = SR(options,selected_tr_labels,selected_tr_points);
predictions=[];
test_labels=[];
while 1
    starting_count1=tic;
    if pointerObs+batch>=size(EvalSetIndices,1) 
        batch=size(EvalSetIndices,1)-pointerObs;
    end
   [XNew,YNew]=getDataInstancesSequential(XFileID,settings.formattingString,settings.delimiter,EvalSetIndices(pointerObs:pointerObs+batch));
   if(size(XNew,1)==0)
       break
   end
   predictions=[predictions;XNew*eigvector];
   test_labels=[test_labels;YNew];
   pointerObs=pointerObs+batch+1;
end
[xx,yy,~,auc] = perfcurve(test_labels, predictions,settings.positiveClass);
end