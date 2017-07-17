function [sampleList,values,Dist,K,updated_sample,updated_class] = MAED_incremental(original_sample,original_sample_class,new_data_point,new_data_point_class,indices_to_remove,D,selectNum,options,warping)
% MAED: Manifold Adaptive Experimental Design  
%
%     sampleList = MAED(fea,selectNum,options)
%   
%     Input:
%
%              fea      - Data matrix. Each row of fea is a sample.
%         selectNum     - The number of samples to select.
%           options     - Struct value in Matlab. The fields in options
%                         that can be set:
%
%                splitLabel    -  logical array with the size as the number
%                                 of samples. If some of the inputs already
%                                 have the label and there is no need
%                                 select these samples, you should set the
%                                 corresponding entry in 'splitLabel' as
%                                 true; (Default: all false)
%
%                      W       -  Affinity matrix. You can either call
%                                 "constructW" to construct the W, or
%                                 construct it by yourself.
%                                 If 'W' is not provided and 'ReguBeta'>0 , 
%                                 MAED will build a k-NN graph with Heat kernel
%                                 weight, where 'k' is a prameter. 
%
%                      k       -  The parameter for k-NN graph (Default is 5)
%                                 If 'W' is provided, this parameter will be
%                                 ignored. 
%
%                  ReguBeta    -  regularization paramter for manifold
%                                 adaptive kernel. 
%
%                  ReguAlpha   -  ridge regularization paramter. Default 0.01
%
%
%     Output:
%
%        sampleList     - The index of the sample which should be labeled.
%
%    Examples:
%
%     See: http://www.zjucadcg.cn/dengcai/Data/ReproduceExp.html#MAED
%
%Reference:
%
%   [1] Deng Cai and Xiaofei He, "Manifold Adaptive Experimental Design for
%   Text Categorization", IEEE Transactions on Knowledge and Data
%   Engineering, vol. 24, no. 4, pp. 707-719, 2012.  
%
%   version 2.0 --Jan/2012
%   version 1.0 --Aug/2008 
%
%   Written by Deng Cai (dengcai AT gmail.com)
%


nSmp = size(original_sample,1);

splitLabel = false(nSmp,1);
if isfield(options,'splitLabel')
    splitLabel = options.splitLabel;
end

%I changed here
%fprintf('Current size of D is %d',size(D,1))
if isempty(indices_to_remove)
    Dist = EuDist2([original_sample;new_data_point],[],0);
    updated_sample=[original_sample;new_data_point];
    updated_class=[original_sample_class;new_data_point_class];
    nSmp=size(updated_sample,1);
elseif isempty(new_data_point)
    Dist = EuDist2(original_sample,[],0);
    updated_sample=original_sample;
    updated_class=original_sample_class;
    nSmp=size(updated_sample,1);
else
    [Dist,updated_sample,updated_class]=EuDist2_incremental(original_sample,original_sample_class,D,indices_to_remove,new_data_point,new_data_point_class,0);
    nSmp=size(updated_sample,1);
end
K = constructKernel_incremental(Dist,options);


if isfield(options,'ReguBeta') && options.ReguBeta > 0
   
    if isfield(options,'W')
        W = options.W;
    else
        if isfield(options,'k')
            Woptions.k = options.k;
        else
            Woptions.k = 0;
        end

        tmpD=Dist;
        Woptions.bLDA=options.bLDA;
        Woptions.t = options.t;
        Woptions.NeighborMode = options.NeighborMode ;
        Woptions.gnd = updated_class ;
        Woptions.WeightMode = options.WeightMode  ;
        %Woptions.t = ceil(length(updated_class)/2);        
        W = constructW(updated_sample,Woptions);
    end
    D = full(sum(W,2));
    L = spdiags(D,0,nSmp,nSmp)-W;
        
    K=(speye(size(K,1))+options.ReguBeta*K*L)\K;
    K = max(K,K');
end

if ~isfield(options,'Method')
    options.Method = 'Seq';
end

ReguAlpha = 0.01;
if isfield(options,'ReguAlpha')
    ReguAlpha = options.ReguAlpha;
end


switch lower(options.Method)
    case {lower('Seq')} 
        [sampleList,values] = MAEDseq(K,selectNum,splitLabel,ReguAlpha);
        if isempty(indices_to_remove)
           updated_sample=updated_sample(sampleList,:);
           updated_class=updated_class(sampleList,:);
           Dist=Dist(sampleList,sampleList);
           K=K(sampleList,sampleList);
           %Dist = EuDist2(updated_sample,[],0);
           %K = constructKernel_incremental(Dist,options);
        end
            
    otherwise
        error('Optimization method does not exist!');
end


