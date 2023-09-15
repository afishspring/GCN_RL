function [InfNei, CurNei] = getNei(i, state, Adj)
%getNei returns the indicator vector of infected and cured neighbors of a
%vertex. Adjacency matrix can be substituted by Beta or Gamma since they
%contain the same adjacency info.
InfSet = (state>0);
CurSet = (state<0);
NeiHd = (Adj(i,:)>0); % Nonzero elements of a row
InfNei = InfSet .* NeiHd;
CurNei = CurSet .* NeiHd;
return
end

