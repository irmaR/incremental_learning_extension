function [Xstandardized] = standardize(X,min,max)
Xstandardized = (X - min )/( max- min);