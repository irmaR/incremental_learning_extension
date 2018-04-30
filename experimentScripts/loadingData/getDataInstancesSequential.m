function [X,Y]=getDataInstancesSequential(fid,formatting,delimiter,indices)
X=[];
Y=[];
for i=1:size(indices,1)
    if indices(i)==0 %skip the header
        continue
    end
    fseek(fid, indices(i), 'bof');
    instance=textscan(fid, formatting, 1, 'delimiter',delimiter);
    if strcmp(instance{1},'T')
        Y(i,:)=1;
    else
        Y(i,:)=2;
    end
    X(i,:)=cell2mat(instance(2:end));
end
end