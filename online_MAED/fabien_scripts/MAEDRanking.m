function [smpRank] = MAEDRanking(candidates, selectNum, model,options)
%Reference:
%
%   [1] Deng Cai and Xiaofei He, "Manifold Adaptive Experimental Design for
%   Text Categorization", IEEE Transactions on Knowledge and Data
%   Engineering, vol. 24, no. 4, pp. 707-719, 2012.
%


fea = candidates.X;
labels = candidates.Y;

nSmp = size(fea,1);

if(~isfield(options,'ReguBeta'))
    options.ReguBeta = .1;
end

if(~isfield(options,'bLDA'))
    options.bLDA = 0;
end

if(~isfield(options,'ReguAlpha'))
    options.ReguAlpha = 0.01;
end

if(isfield(options,'supervised'))
    options.gnd = labels;
end

[K,Dist,options] = constructKernel(fea,[],options);

if isfield(options,'ReguBeta') && options.ReguBeta > 0
    if isfield(options,'W')
        W = options.W;
    else
        if isfield(options,'k')
            Woptions.k = options.k;
        else
            Woptions.k = 5;
        end
        tmpD = Dist;
        Woptions.t = mean(mean(tmpD));
        if isfield(options,'gnd')
            Woptions.WeightMode = 'HeatKernel';
            Woptions.NeighborMode='Supervised';
            Woptions.bLDA=options.bLDA;
            Woptions.gnd=options.gnd;
        end
        W = constructW(fea,Woptions);
    end
    D = full(sum(W,2));
    L = spdiags(D,0,nSmp,nSmp)-W;
    K = (speye(size(K,1))+options.ReguBeta*K*L)\K;
    K = max(K,K');
end

splitCandi = true(size(K,2),1);
smpRank = zeros(selectNum,1);

for sel = 1:selectNum
    DValue = sum(K(:,splitCandi).^2,1)./(diag(K(splitCandi,splitCandi))'+options.ReguAlpha);
    [~,idx] = max(DValue);
    CandiIdx = find(splitCandi);
    smpRank(sel) = CandiIdx(idx);
    splitCandi(CandiIdx(idx)) = false;
    K = K - (K(:,CandiIdx(idx))*K(CandiIdx(idx),:))/(K(CandiIdx(idx),CandiIdx(idx))+options.ReguAlpha);
end

end

