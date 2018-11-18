function [x,f,g] = bb(fn,x0,lrate,verbose)
%BB Barzilai & Borwein's two-point step size gradient descent algorithm.
%
% function [x,fx,exitflag] = bb(fn,x0,lrate)
%
% INPUT:
%   fn:      handle to the objective function to minimize
%   x0:      initial point
%   lrate:   initial learning rate
%   verbose: if true, print verbose information
%
% OUTPUT:
%   x:     the point which achieves the optimal value
%   f:     the optimal objective value
%   g:     the gradient at the optimal point
%

% By Wei Tang (wtang@cs.utexas.edu)

if nargin < 3,
    lrate = 1e-5;                       % default learning rate
    verbose = false;
elseif nargin < 4,
    verbose = false;
end

tol = 1e-5;                             % stopping criteria
x = x0;                                 % current x
[f,g] = fn(x);                          % current f and g
i = 1;
while true
    if i > 1
        if norm(dg,'fro')/f < tol,
            fprintf('Converged!\n');
            break;
        end
%         lrate = max((dx'*dg)/(dg'*dg),(dx'*dx)/(dx'*dg));
        if mod(i,2) == 1,
            lrate = (dx'*dg)/(dg'*dg);
        else
            lrate = (dx'*dx)/(dx'*dg);
        end
    end
    if verbose,
        fprintf('iter %d: norm(g) = %.2f, f = %.2f\n',i,norm(g,'fro'),f);
    end
    ox = x; og = g;                     % save the current x and g
    x = ox-lrate*og;                    % gradient descent
    [f,g] = fn(x);
    dx = x(:)-ox(:);
    dg = g(:)-og(:);
    i = i+1;
end

end % bb function