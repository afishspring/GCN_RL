function [ IndBall ] = ballExpl(c, Dist, r)
%BALLEXPL extacts the indices of vertices fall into the ball with radius r
% at the center c, from the graph with weighted adjacency matrix Adjw.
IndBall = find(Dist(c,:) <= r);
end