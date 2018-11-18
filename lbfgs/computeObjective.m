function f=computeObjective(c)
% c: other constants used to evaluate f and df
%    c.UserGroup: (matrix)
%    c.UserUser: (matrix)
%    c.linkWt: (scalar)
%    c.reg: (scalar) regularization parameter
%    c.UserFactors: (matrix)
%    c.GroupFactors: (matrix)
%    c.L1: d by d dimension
%    c.L2: d by d dimension
    tic
    f = norm(c.UserGroup, 'fro')^2;
    f = f - 2 * sum(dot(c.UserFactors * c.L1, c.UserGroup * c.GroupFactors));
    f = f + norm(c.UserFactors * c.L1 * c.GroupFactors', 'fro')^2;
    f = f + c.linkWt * norm(c.UserUser, 'fro')^2;
    f = f - 2 * c.linkWt * sum(dot(c.UserFactors * c.L2, c.UserUser * c.UserFactors));
    f = f + c.linkWt * norm(c.UserFactors * c.L2 * c.UserFactors', 'fro')^2;
    
    f = f + c.reg * (norm(c.UserFactors, 'fro')^2 + norm(c.GroupFactors, 'fro')^2 + norm(c.L1, 'fro')^2 + norm(c.L2, 'fro')^2);
    toc
    fprintf(1,'Computed objective: %d\n', f);
end