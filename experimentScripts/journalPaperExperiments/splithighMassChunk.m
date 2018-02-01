
function[Data,Class]=splithighMassChunk(trainChunk)
   M = csvread(trainChunk,1,0);
   Class=M(:,1);
   Data=M(:,2:end);
end