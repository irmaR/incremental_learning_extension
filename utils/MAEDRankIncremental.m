function [model] = MAEDRankIncremental(model,XNew,YNew,numSamples,options)
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

model.D=updateEuclidDist(model.X,model.D,XNew);
model.K = constructKernelIncremental(model.D,options); %---------- BRING
%add new points to the sample
indexCutoff=size(model.X,1);
model.X=[model.X;XNew];
model.Y=[model.Y;YNew];
sizeAugmentedSample=size(model.X,1);
nSmp = size(model.X,1);
splitLabel = false(nSmp,1);
if isfield(options,'splitLabel')
    splitLabel = options.splitLabel;
end

if isfield(options,'warping')
    options.gnd=model.Y;
end

if isfield(options,'ReguBeta') && options.ReguBeta > 0
    if isfield(options,'W')
        W = options.W;
    else
        if isfield(options,'k')
            Woptions.k = options.k;
        else
            Woptions.k = 5;
        end
        
        tmpD = model.D;
        Woptions.t = mean(mean(tmpD));
        if isfield(options,'gnd')
            Woptions.WeightMode = 'HeatKernel';
            Woptions.NeighborMode='Supervised';
            Woptions.bLDA=options.bLDA;
            Woptions.gnd=options.gnd;
        end
        W = constructW(model.X,Woptions);
    end
    D = full(sum(W,2));
    L = spdiags(D,0,nSmp,nSmp)-W;
    K=(speye(size(model.K,1))+options.ReguBeta*model.K*L)\model.K;
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
        [sampleList,values] = MAEDseq(K,numSamples,splitLabel,ReguAlpha);
        model.X=model.X(sampleList,:);
        model.Y=model.Y(sampleList,:);
        model.D=model.D(sampleList,sampleList);
        model.K=model.K(sampleList,sampleList);
        model.betas=values;
        nrRemoved=size(find(sampleList<=indexCutoff),1);
        model.percentageRemoved=(nrRemoved/sizeAugmentedSample)*100;
    otherwise
        error('Optimization method does not exist!');
end
end



