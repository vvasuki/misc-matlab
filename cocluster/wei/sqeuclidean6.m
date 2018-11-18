function D = sqeuclidean6(X, Y, I, C, W)
% SQEUCLIDEAN6 Calculate point to cluster centroid distances by squared
% Euclidean distance. (base 6)

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: sqeuclidean6.m,v 1.2 2008/04/19 01:20:47 wtang Exp $

if (size(W,1) ~= size(X,1)) || (size(W,2) ~= size(X,2)) ...
		|| (size(X,2) ~= size(C,2))
	error('Dimensions of the weight and input matrices do not match');
end

% [n,p] = size(X);
% nclusts = size(C,1);
% D = zeros(n,nclusts);
% 
% for i = 1:nclusts
% 	D(:,i) = W(:,1).*(X(:,1) - Y(:,I(1)) - C(i,1)).^2;
% 	for j = 2:p
% 		D(:,i) = D(:,i) + W(:,j).*(X(:,j) - Y(:,I(j)) - C(i,j)).^2;
% 	end
% end
D = sqeuclidean6c(X, Y, I, C, W);
end % sqeuclidean6 function