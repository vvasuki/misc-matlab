function [f,df] = evalL1(L1, c)
% L1: current L1 matrix
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
c.L1 = L1;

% Compute the objective function value.
f=computeObjective(c);

% Compute the gradient w.r.t. L1
df = -2 * c.GroupFactors' * c.UserGroup' * c.UserFactors + 2 * c.GroupFactors' * c.GroupFactors * c.L1 * c.UserFactors' * c.UserFactors;
df = df + c.reg * 2 * c.L1;
end
