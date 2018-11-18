function D = idivergence6(X, C, I, V, W)
% IDIVERGENCE6 Calculate point to cluster centroid distances by I
% divergence. (base 6)

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: idivergence6.m,v 1.2 2008/04/19 01:20:47 wtang Exp $

if (size(W,1) ~= size(X,1)) || (size(W,2) ~= size(X,2)) ...
		|| (size(C,1) ~= size(X,1)) || (size(X,2) ~= size(V,2))
	error('Dimensions of the weight and input matrices do not match');
end

% [n,p] = size(X);
% nclusts = size(V,1);
% D = zeros(n,nclusts);
% 
% for i = 1:nclusts
% 	D(:,i) = W(:,1).*(C(:,I(1))*V(i,1) - X(:,1)*log(V(i,1)));
% 	for j = 2:p
% 		D(:,i) = D(:,i) + W(:,j).*(C(:,I(j))*V(i,j) - X(:,j)*log(V(i,j)));
% 	end
% end
D = idivergence6c(X, C, I, V, W);
end % idivergence6 function