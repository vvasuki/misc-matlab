function [P,L] = MatFactBB(A,P,L,gamma,dense)

%MatFactBB performs the linked matrix factorization using BB algorithm
%directly.
%
% function [P,L] = MatFactBB(A,P,L,gamma,dense) 
%
% Input:
%   A: m by 1 cell variable, each cell A{i} represents a slice 
%      (similarity matrix)
%   P: n by f initial matrix factor, where n is the dimension of
%      A{i}, f is the number of factors
%   L: m by 1 cell variable, each cell L{i} represents f
%      by f symmetric matrix factor
%   gamma: scalar variable, the regularization parameter
%   dense: scalar variable, true for A are dense matrices, false
%      otherwise (default: false)
%
% Output:
%   P: n by f resultant matrix factor
%   L: m by 1 cell variable, each cell L{i} is the resultant f by f
%      symmetric matrix factor
% 
% Example: suppose we have 5 slices, each has n = 100 and f = 20;
%
% % initialize initP and initL
% initP = rand(100,20);
% L = rand(20);
% L = (L+L')/2;  % make L symmetric
% initL(1:5) = {L};
%
% % invoke the matrix factorizatio code
% [P,L] = MatFactBB(A,initP,initL,0.35);
%

if nargin < 5
    dense = true;
end

consts.A = A;
consts.r = gamma;
consts.n = size(P);
consts.w = dense;
fprintf('\tminimizing P and Lambda matrices ... \n');
[x,fx] = bb(@(x)evalF(x,consts),[P;cell2mat(L)'],1e-9);
P = x(1:consts.n(1),:);
L = mat2cell(x(consts.n(1)+1:end,:),repmat(consts.n(2),1,length(A)),consts.n(2));
fprintf('obj = %.4f\n', fx);
end % MatFactLBFGS function
