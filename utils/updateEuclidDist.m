function [D] = updateEuclidDist(X,D,XNew)
%This function incrementally updates the Euclid distance matrix D with new
%points XNew
if isempty(D)
    D=EuDist2(XNew,[]);
else
    D1=EuDist2(XNew,X);
    D2=EuDist2(XNew,[]);
    D=[D,D1';D1,D2];

%    Test=EuDist2([X;XNew],[]);
%    assert(isequal(D,Test));
end

