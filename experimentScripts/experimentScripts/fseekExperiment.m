fid=fopen('/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/test.csv');
arrayofOffsets=[];
lineNumber=1;
id=ftell(fid);
runtime=tic;
while 1
    if mod(lineNumber,1000000)==0
        fprintf('Reached %d in %d seconds',lineNumber,toc(runtime))
    end
    tline = fgetl(fid);
    if(tline==-1)
        break
    end
    arrayofOffsets(lineNumber,:)=id; 
    id=ftell(fid);
    lineNumber=lineNumber+1;
end
save('/Users/irma/Documents/MATLAB/DATA/HighMassPhysics/DummyoffsetIndices.mat','arrayofOffsets');
%test see if we are getting the right indices
for k=1:size(arrayofOffsets)
   fseek(fid, arrayofOffsets(k), 'bof');
   %tline = fgetl(fid)
   out=cell2mat(textscan(fid, '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f', 1, 'delimiter',','));
end
fseek(fid, arrayofOffsets(3), 'bof')
out=cell2mat(textscan(fid, '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f', 1, 'delimiter',','))
fclose(fid); 