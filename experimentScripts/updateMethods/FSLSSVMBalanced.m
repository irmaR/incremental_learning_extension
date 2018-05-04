function [Xs,Ys]=FSLSSVMBalanced(XObs,Yobs,Nc,options)
Xs=XObs(1:Nc,:);
Ys=Yobs(1:Nc,:);
crit_old=-inf;
for tel=1:200 %we are really doing them a favor, but it's so slow
    Xsp=Xs; Ysp=Ys;
    S=ceil(size(XObs,1)*rand(1));
    Sc=ceil(Nc*rand(1));
    Xs(Sc,:) = XObs(S,:);
    Ys(Sc,:) = Yobs(S);
    Ncc=Nc;
    crit = kentropy(Xs,options.kernel_type, options.kernel);
    if crit <= crit_old,
        crit = crit_old;
        Xs=Xsp;
        Ys=Ysp;
    else
        crit_old = crit;
    end
end
end