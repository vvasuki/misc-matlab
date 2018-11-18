function [P,L] = MatFactLBFGSAMsemi(A,W,P,L,S,gamma,beta)

%MatFactLBFGSAM performs the linked matrix factorization using Alternating
%Minimization with L-BFGS algorithm.
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
%   S: n by n smoothness matrix
%   gamma: scalar variable, the regularization parameter
%   beta:  scalar variable, the smoothness regularization parameter
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
% W = cell(length(A),1);
% for i = 1:length(A)
%   W{i} = spones(size(A{i})); % only nonzero entries are considered as known
% end
%
% % invoke the matrix factorizatio code
% [P,L] = MatFactLBFGSAM(A,W,initP,initL,0.35);
%

for i = 1:length(W)
    if ~islogical(W{i})
        W{i} = logical(W{i});
    end
end

opts = lbfgs_options('iprint',-1,'maxits',1000,'factr',1e5,'cb',@lbfgs_null_cb);
%opts = lbfgs_options('iprint',-1,'maxits',1000,'factr',1e5,'cb',@test_callback);
consts.A = A;
consts.w = W;
consts.r = gamma;
consts.n = size(P);
consts.S = S;
consts.b = beta;
tol = 1e-1;                             % tolerence
maxi = 300;                             % maximum number of iterations
for t = 1:maxi
	fprintf('iteration (%d):\n',t);
    % minimize wrt P
	fprintf('\tminimizing with respect to P ... ');
    consts.L = L;
    [P,fx] = lbfgs(@(x)evalPsemi(x,consts),P,[],[],[],opts);
	fprintf('done!\n');
    
    % minimize wrt L
    for j = 1:length(A)
        fprintf('\tminimizing with respect to L%d ... ',j);
        consts.i = j;
        consts.P = P;
        [L{j},fx] = lbfgs(@(x)evalLsemi(x,consts),L{j},[],[],[],opts);
        consts.L{j} = L{j};
		fprintf('done!\n');
    end
    fprintf('objective function value: %.3f\n', fx);
	if t > 1
		if abs(oldfx-fx) < abs(oldfx)*tol || isnan(fx)
			disp('converged!');
			break;
		end
	end
    oldfx = fx;
end

end % MatFactLBFGSAM function
