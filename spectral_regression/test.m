fea = rand(50,70);
gnd = [ones(10,1);ones(15,1)*2;ones(10,1)*3;ones(15,1)*4];
options = [];
options.gnd = gnd;
options.ReguAlpha = 0.01;
options.ReguType = 'Ridge';
[eigvector] = SR_caller(options, fea);

if size(eigvector,1) == size(fea,2) + 1
    Y = [fea ones(size(fea,1),1)]*eigvector;  % Y is samples in the SR subspace
else
    Y = fea*eigvector;  % Y is samples in the SR subspace
end