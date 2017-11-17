function [results]=runExperiment(settings,method)
switch lower(method)
    case {lower('iSRDA_unb')}
        results=incremental(training_data,training_class,test_data,test_class,reguAlphaParams,reguBetaParams,kernel_params,nr_samples,interval,batch_size,report_points,data_limit,'incr',r,warping,blda,k,WeightMode,NeighborMode,@srdaInference);
    case {lower('iSRDA')}
        results=iSRKDA(settings,@srdaInference);
    case {lower('iSRKDA')}
        results=iSRKDA(settings,@srkdaInference);    
    case {lower('SRDA')}
        results=bSRDKA(settings,@srdaInference);
    case {lower('SRKDA')}
        results=bSRDKA(settings,@srkdaInference);
    case {lower('iSRDASeq')}
        results=iSRKDASequential(settings,@srdaInference);
    case {lower('iSRKDASeq')}
        results=iSRKDASequential(settings,@srkdaInference);    
    case {lower('SRDASeq')}
        results=bSRKDASequential(settings,@srdaInference);
    case {lower('SRKDASeq')}
        results=bSRKDASequential(settings,@srkdaInference);
    case {lower('random')}
        results=random(settings,@srkdaInference);
    case {lower('randomSeq')}
        results=randomSequential(settings,@srkdaInference);
    case {lower('lssvm')}
        results=incrementalLSSVM(settings);
end
end




