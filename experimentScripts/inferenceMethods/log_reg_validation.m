function [area]=log_reg_validation(kernel,selected_tr_points,selected_tr_labels,settings,dummy,options)
batch=settings.read_size_test;
pointerObs=1;
%mdl = fitglm(selected_tr_labels,selected_tr_labels,'Distribution','binomial','Link','logit')
mdl=mnrfit(selected_tr_points,selected_tr_labels);
predictions=[];
test_labels=[];
while 1
    starting_count1=tic;
    if pointerObs+batch>=size(settings.indicesOffsetValidation,1) 
        batch=size(settings.indicesOffsetValidation,1)-pointerObs;
    end
   [XNew,YNew]=getDataInstancesSequential(settings.XTrainFileID,settings.formattingString,settings.delimiter,settings.indicesOffsetTest(pointerObs:pointerObs+batch));
   if(size(XNew,1)==0)
       break
   end
   predictions=[predictions;mnrval(mdl,XNew)];
   test_labels=[test_labels;YNew];
   pointerObs=pointerObs+batch+1;
end
[X,Y,T,area] = perfcurve(test_labels,predictions(:,1),'1');
end