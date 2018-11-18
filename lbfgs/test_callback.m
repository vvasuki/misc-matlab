function [stop,user_data] = test_callback(x,iter,state,user_data,opts)
% TEST_CALLBACK  Example callback for the lbfgs wrapper.
%
% [stop,ud] = lbfgs_null_cb(x,iter,state,ud,opts)
%
% Copyright 2005-2006 Liam Stewart
% See COPYING for license.

switch state
    case 'init'
        user_data.f = zeros(1,opts.maxits);
        user_data.x = zeros(size(x,1),size(x,2)*opts.maxits);
        user_data.its = 0;
    case 'iter'
        user_data.f(iter.it) = iter.f;
        user_data.x(:,iter.it*size(x,2)+1:(iter.it+1)*size(x,2)) = x;
        user_data.its = user_data.its + 1;

        fprintf('Iteration %4d\tf: %-8.6g\n', iter.it, iter.f);
    case 'done'
        user_data.f = user_data.f(1:user_data.its);
        user_data.x = user_data.x(:,user_data.its*size(x,2)+1:(user_data.its+1)*size(x,2));
end

stop = 0;
