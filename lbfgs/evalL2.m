function [f,df] = evalL2(L2, c)
% L2: current L2 matrix
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
c.L2 = L2;

% Compute the objective function value.
f=computeObjective(c);

% Compute the gradient w.r.t. L2
df = -2 * c.UserFactors' * c.UserUser * c.UserFactors + 2 * c.UserFactors' * c.UserFactors * c.L2 * c.UserFactors' * c.UserFactors;
df = df + c.reg * 2 * c.L2;
end
