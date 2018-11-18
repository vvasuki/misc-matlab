function D = idivergence(X, C, V, W)
% IDIVERGENCE Calculate point to cluster centroid distances by I
% divergence.

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: idivergence.m,v 1.2 2008/04/19 01:20:47 wtang Exp $

if (size(W,1) ~= size(X,1)) || (size(W,2) ~= size(X,2)) ...
		|| (size(C,1) ~= size(X,1)) || (size(C,2) ~= size(X,2)) ...
		|| (size(X,2) ~= size(V,2))
	error('Dimensions of the weight and input matrices do not match');
end

% [n,p] = size(X);
% nclusts = size(V,1);
% D = zeros(n,nclusts);
% 
% for i = 1:nclusts
% 	D(:,i) = W(:,1).*(C(:,1)*V(i,1) - X(:,1)*log(V(i,1)));
% 	for j = 2:p
% 		D(:,i) = D(:,i) + W(:,j).*(C(:,j)*V(i,j) - X(:,j)*log(V(i,j)));
% 	end
% end
D = idivergencec(X, C, V, W);
end % idivergence function
