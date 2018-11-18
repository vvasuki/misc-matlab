function [f,df] = evalUserFactors(UserFactors, c)
% UserFactors: current UserFactors matrix
% c: other constants used to evaluate f and df
%    c.UserGroup: (matrix)
%    c.UserUser: (matrix)
%    c.linkWt: (scalar)
%    c.reg: (scalar) regularization parameter
%    c.UserFactors: (matrix)
%    c.GroupFactors: (matrix)
%    c.L1: d by d dimension
%    c.L2: d by d dimension

if nargin < 2
    error('Function needs two input arguments!\n');
end

% calculate f and df
c.UserFactors = UserFactors;

% Compute the objective function value.
f=computeObjective(c);

% Compute the gradient w.r.t. UserFactors
df = -2 * c.UserGroup * c.GroupFactors * c.L1 + 2 * c.UserFactors * c.L1 * c.GroupFactors'*c.GroupFactors*c.L1 - 4 * c.UserUser * c.UserFactors * c.L2 + 4 * c.UserFactors * c.L2 * c.UserFactors' * c.UserFactors * c.L2;
df = df + c.reg * 2 * c.UserFactors;
end
