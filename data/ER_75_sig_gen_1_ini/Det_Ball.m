function [ result ] = Det_Ball(Dist, State)
%DET_BALL Summary of this function goes here

Infected_Nodes = find(State ~=0);
infCnt = sum((State>0));
threshold = length(Infected_Nodes) / size(Dist, 1);

N = size(Dist, 1);
infection_boolean=zeros(N,1);

for i = 1 : N
    %indices=Neigh_Explore(i,Wei_Adj_mat,10)
    indices = ballExpl(i,Dist,300*infCnt/N);

    c = intersect(indices,Infected_Nodes);
    if (length(c)/length(indices) >=threshold)
        infection_boolean(i)=1;
        result = 'Epidemic';
        break
    end
    %fprintf('Looping vertex%i, infection_boolean(i) = % i\n', i, infection_boolean(i));
end

if any(infection_boolean(:) ~=0)
   result = 'Epidemic';
else
   result = 'Random Illness';
end

end