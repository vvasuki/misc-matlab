function [P,L] = MatFactLBFGS(A,W,P,L,gamma)

%MatFactLBFGS performs the linked matrix factorization using L-BFGS
%algorithm directly.
%
% Input:
%   A: m by 1 cell variable, each cell A{i} represents a slice 
%      (similarity matrix)
%   W: m by 1 cell variable, each cell W{i} indicates the known entries in
%      the corresponding A{i}
%   P: n by f initial matrix factor, where n is the dimension of
%      A{i}, f is the number of factors
%   L: m by 1 cell variable, each cell L{i} represents f
%      by f symmetric matrix factor
%   gamma: scalar variable, the regularization parameter
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
% [P,L] = MatFactLBFGS(A,initP,initL,0.35);
%

for i = 1:length(W)
    if ~islogical(W{i})
        W{i} = logical(W{i});
    end
end

opts = lbfgs_options('iprint',-1,'maxits',1000,'factr',1e7,'cb',@lbfgs_null_cb);
consts.A = A;
consts.w = W;
consts.r = gamma;
consts.n = size(P);
fprintf('\tminimizing P and Lambda matrices all together ... \n');
[x,fx] = lbfgs(@(x)evalF(x,consts),[P;cell2mat(L)'],[],[],[],opts);
P = x(1:consts.n(1),:);
L = mat2cell(x(consts.n(1)+1:end,:),repmat(consts.n(2),1,length(A)),consts.n(2));
fprintf('obj = %.4f\n', fx);
end % MatFactLBFGS function
