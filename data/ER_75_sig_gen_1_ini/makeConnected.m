function [ Adj ] = makeConnected( A )
%MAKECONNECTED select the largest component
%   Detailed explanation goes here
[~,~,members] = networkComponents(A);
Vertex = members{1,1};
Adj = zeros(size(Vertex,2));
for i=1:size(Vertex,2)
    for j=1:size(Vertex,2)
        Adj(i,j) = A(Vertex(i),Vertex(j));
    end
end
end

