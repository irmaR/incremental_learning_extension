function [area]=srkdaInference(kernel,selected_tr_points,selected_tr_labels,test_data,test_class,options)
%if we only have one class, return area=0
   if length(unique(selected_tr_labels))==1
       fprintf('only one label selected!\n')
       unique(selected_tr_labels)
       fprintf('\n')
       area=NaN;
       return
   end
   options.Kernel=1;
   options.ReguType = 'Ridge';
   options.gnd=selected_tr_labels;  
   [eigvector, elapseKSR] = KSR_caller(options, kernel);
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
   Ktest = constructKernel(test_data, selected_tr_points, options);
   Yhat = Ktest*eigvector;
   if sum(isnan(Yhat))~=0
       area=0;
   else
   [X,Y,T,area] = perfcurve(test_class,Yhat,'1');
   end
   options.Kernel=0;
end