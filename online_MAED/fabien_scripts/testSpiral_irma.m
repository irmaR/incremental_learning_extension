%function testSpiral2()
close all;
load('./twoSpirals.mat');

% not displaying the label clc
figure;
class1=find(Y==0);
class2=find(Y==1);
plot(X(class1,1), X(class1,2), 'r.');  hold on;
plot(X(class2,1), X(class2,2), 'b.');  hold on;

% displaying the label Y 
%for i=1:length(X)
%    plot(X(i,1), X(i,2), '.', 'Color', [Y(i)./2  Y(i)./2 1]); hold on;
%end

options.t = 1;

s = RandStream('mt19937ar','Seed',1);    
idx = randperm(s,numel(Y));
X = X(idx,:);
Y = Y(idx,:);

%[model] = incrementalLearn(X, Y, options);
[~,model]=MAED_incremental(X,Y,25,100,options);


figure;
plot(X(:,1), X(:,2), '.');  hold on;
for i=1:25
    class1=find(model.Y==0);
    class2=find(model.Y==1);
    plot(model.X(class1,1), model.X(class1,2),'rx','MarkerSize',12); hold on;
    plot(model.X(class2,1), model.X(class2,2),'gx','MarkerSize',12); hold on;
end