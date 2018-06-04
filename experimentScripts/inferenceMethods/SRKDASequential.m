function [auc]=SRKDASequential(selected_tr_points,selected_tr_labels,settings,options,EvalSetIndices,XFileID)
batch=settings.read_size_test;
pointerObs=1;
options.ReguType = 'Ridge';
options.gnd = selected_tr_labels;
options.Kernel=1;
K = constructKernel(selected_tr_points, [], options);
[eigvector, elapseKSR] = KSR(options, selected_tr_labels, K);
if isempty(eigvector)
    options=rmfield(options,'gnd');
    Woptions.gnd = selected_tr_labels ;
    Woptions.t = options.t;
    Woptions.k=options.k;
    Woptions.NeighborMode = options.NeighborMode ;
    Woptions.WeightMode=options.WeightMode;
    W = constructW(selected_tr_points,Woptions);
    options.W=W;
    options.ReducedDim = 1;
    [eigvector, elapseKSR] = KSR_caller(options, kernel);
    options=rmfield(options,'W');
    options=rmfield(options,'ReducedDim');
end


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
    Ktest = constructKernel(XNew, selected_tr_points, options);
    Yhat = Ktest*eigvector;
    predictions=[predictions;Yhat];
    test_labels=[test_labels;YNew];
    pointerObs=pointerObs+batch+1;
end
[xx,yy,~,auc] = perfcurve(test_labels, predictions,settings.positiveClass);
end