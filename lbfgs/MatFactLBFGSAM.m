function [UserFactors, GroupFactors, L1, L2] = MatFactLBFGSAM(UserUser, UserGroup, UserFactors, GroupFactors, L1, L2, linkWt, reg)
%MatFactLBFGSAM performs the linked matrix factorization using Alternating
%Minimization with L-BFGS algorithm.
%
% Input:
%   UserUser, UserGroup
%   UserFactors, GroupFactors
%   L1, L2: Symmetric matrices (f by f)
%   linkWt: scalar variable
%   reg: scalar variable, the regularization parameter
%
% Output:
%   UserFactors, GroupFactors
%   L1, L2: Symmetric matrices (f by f)
%

opts = lbfgs_options('iprint',-1,'maxits',1000,'factr',1e5,'cb',@lbfgs_null_cb);
c.UserUser = UserUser;
c.UserGroup = UserGroup;
c.UserFactors = UserFactors;
c.GroupFactors = GroupFactors;
c.L1 = L1;
c.L2 = L2;
c.linkWt = linkWt;
c.reg = reg;
tol = 1e-1;                             % tolerence
maxi = 300;                             % maximum number of iterations
for t = 1:maxi
    fprintf('iteration (%d):\n',t);
    fprintf('\tminimizing with respect to UserFactors ... ');
    [c.UserFactors,obj] = lbfgs(@(x)evalUserFactors(x,c),c.UserFactors,[],[],[],opts);
    fprintf('done!\n');
    
    fprintf('\tminimizing with respect to GroupFactors ... ');
    [c.GroupFactors,obj] = lbfgs(@(x)evalGroupFactors(x,c),c.GroupFactors,[],[],[],opts);
    fprintf('done!\n');
    
    fprintf('\tminimizing with respect to L1 ... ');
    [c.L1,obj] = lbfgs(@(x)evalL1(x,c),c.L1,[],[],[],opts);
    fprintf('done!\n');
    
    fprintf('\tminimizing with respect to L2 ... ');
    [c.L2,obj] = lbfgs(@(x)evalL2(x,c),c.L2,[],[],[],opts);
    fprintf('done!\n');
    
    fprintf('objective function value: %.3f\n', obj);
    if t > 1
        if oldobj-obj < oldobj*tol
            disp('converged!');
            break;
        end
    end
    oldobj = obj;
end

end % MatFactLBFGSAM function
