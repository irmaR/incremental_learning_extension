function [Xstandardized,minO,maxO] = standardizeX(X)
% Standardize X using 0-1 scaling
minO=min(X(:));
maxO=max(X(:));
Xstandardized = (X - minO )/( maxO- minO);

