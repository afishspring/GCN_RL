function [ result ] = Det_Den(Dist, State)
%DET_Den Detection with Ball Density Algorithm

Infected_Nodes = find(State ~=0);
infCnt = sum((State>0));
Beta=3;

% Algorithm
N = size(Dist, 1);
Nodes_indices = 1:N;
infection_boolean=zeros(N,1);

for i = 1 : N
    indices = ballExpl(i,Dist,300*infCnt/N);
    c = intersect(indices,Infected_Nodes);
    if ~isempty(indices)
        inside_density = length(c)/length(indices);
    else
        inside_density=0;
    end

    indices_prime = setdiff(Nodes_indices,indices);
    c_prime = intersect(indices_prime,Infected_Nodes);
    if ~isempty(c_prime)
        outside_density= length(c_prime)/length(indices_prime);
        relative_denisty = inside_density/outside_density;
        if (relative_denisty >Beta)
            infection_boolean(i) = 1;
            result = 'Epidemic';
            break
        end
    elseif inside_density ~=0
        infection_boolean(i) = 1;
        result = 'Epidemic';
    end
    
        
end

if any(infection_boolean(:) ~=0)
    result = 'Epidemic';
else
    result = 'Random Illness';
end

end