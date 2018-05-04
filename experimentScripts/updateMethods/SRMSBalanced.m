function [Xs,Ys]=SRMSBalanced(XObs,YObs,N,options)
r=0;
verbose = false;
[repInd,C] = smrs(XObs',options.alpha,r,verbose);
indices=repInd';
Xs=XObs(indices,:);
Ys=YObs(indices,:);
end