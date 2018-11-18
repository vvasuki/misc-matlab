function [f,df] = evalGroupFactors(GroupFactors, c)
% GroupFactors: current GroupFactors matrix
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
    error('Function evalGroupFactors needs two input arguments!\n');
end

% calculate f and df
c.GroupFactors = GroupFactors;

% Compute the objective function value.
f=computeObjective(c);

% Compute the gradient w.r.t. GroupFactors
df = -2 * c.UserGroup' * c.UserFactors * c.L1 + 2 * c.GroupFactors * c.L1 * c.UserFactors' * c.UserFactors * c.L1;
df = df + c.reg * 2 * c.GroupFactors;
end
