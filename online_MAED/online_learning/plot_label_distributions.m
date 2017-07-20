function []=plot_label_distributions(label_distr,output)
% plot_label_distributions: plotting label distributions that were
% collected during online learning. 
%  
% Input:
%   label_distr               - is a structure array where each cell contains the count of samples in the sample.
% each cell (assuming here we deal with binary classification problem) is
% of the following form:
%               class1 count1
%               class2 count2
%   output - path where image is going to be saved
x=(1:size(label_distr,2));
y1=[];
y2=[];
for i=1:size(label_distr,2)
  entry=label_distr{i};
  y1(i)=entry(1,2);
  y2(i)=entry(2,2);
end

fig = figure('visible', 'off');
colorVec = hsv(2);
plot(x,y1,'LineWidth',2,'Color',colorVec(1,:)); hold on;
legendInfo{1} = ['class1'];
plot(x,y2,'LineWidth',2,'Color',colorVec(2,:)); hold on;
legendInfo{2} = ['class 2'];
legend(legendInfo);
hold off;
if exist(sprintf('%s/label_distr.jpg',output), 'file')==2
  delete(sprintf('%s/label_distr.jpg',output));
end

saveas(fig,sprintf('%s/label_distr.jpg',output))


end