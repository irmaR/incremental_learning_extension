function [X,Y]=getDataInstancesSequential(fid,formatting,delimiter,indices)
X=[];
Y=[];
for i=1:size(indices,1)
    if indices(i)==0 %skip the header
        continue
    end
    fseek(fid, indices(i), 'bof');
    instance=textscan(fid, formatting, 1, 'delimiter',delimiter);
    out=cell2mat(instance);
    Y(i,:)=out(1);
    X(i,:)=out(2:end);
end
end