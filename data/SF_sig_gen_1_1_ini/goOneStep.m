function [ tmpState ] = goOneStep(state, beta)
%GOONESTEP Let the epidemic evolve for one time step.
% If the infection count hits the stub, log the state vector.
n = length(state);
RanM = rand(n);
tmpState = state;

for i = 1:n        
    if state(i) == 0
        [InfNei, ~] = getNei(i, state, beta);
        ActI = bsxfun(@le, RanM(i,:), beta(i,:)) .* InfNei;
        if any(ActI) ~= 0
            tmpState(i) = 1;
        end
    else tmpState(i) = state(i)+1;
    end
end

return
end

