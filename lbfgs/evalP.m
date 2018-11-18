function [f,df] = evalP(x,c)
% x: the P matrix (n by f)
% c: other constants used to evaluate f and df
%    c.A: (cell) m adjacency matrices, each in dimension n by n
%    c.w: (cell) 0/1 weight matrix
%    c.r: (scalar) regularization parameter
%    c.n: (matrix) size of the P matrix, dimension 2 by 1
%    c.L: (cell) m Lambda matrices, each in dimension f by f

if nargin < 2
    error('Function evalP needs two input arguments!\n');
end

% calculate f and df
f = 0;                                  % objective value
df = zeros(size(x));                    % gradient of P matrix

for j = 1:length(c.A)
    if nnz(c.w{j})/numel(c.w{j})==1
        f = f+norm(c.A{j},'fro')^2;
        tmp = x*c.L{j};
        f = f-2*sum(dot(tmp,c.A{j}*x));
		df = df-2*c.A{j}*tmp+2*tmp*(x'*tmp);
        tmp = x'*tmp;
        f = f+sum(dot(tmp',tmp));
		f = f+c.r*norm(c.L{j},'fro')^2;
    else
		[I,J] = find(c.w{j});
		tmp = x*c.L{j};
		residual = c.A{j}(c.w{j})-dotp(tmp,x,I,J);
		f = f+norm(residual)^2;
		f = f+c.r*norm(c.L{j},'fro')^2;
		df = df-2*sparse(I,J,residual,c.n(1),c.n(1))*tmp;
    end
end
% for j = 1:length(c.A)
%     if nnz(c.w{j})/numel(c.w{j})<=0.5 % when W is very sparse
% 		[I,J] = find(c.w{j});
% 		tmp = x*c.L{j};
% 		residual = c.A{j}(c.w{j})-dotp(tmp,x,I,J);
% 		f = f+norm(residual)^2;
% 		f = f+c.r*norm(c.L{j},'fro')^2;
% 		df = df-2*sparse(I,J,residual,c.n(1),c.n(1))*tmp;
%     else % when W is very dense
%         f = f+norm(c.A{j},'fro')^2;
%         tmp = x*c.L{j};
%         f = f-2*sum(dot(tmp,c.A{j}*x));
% 		df = df-2*c.A{j}*tmp+2*tmp*(x'*tmp);
%         tmp = x'*tmp;
%         f = f+sum(dot(tmp',tmp));
% 		f = f+c.r*norm(c.L{j},'fro')^2;
%         if nnz(c.w{j})/numel(c.w{j})<1 % W is not totally dense
%             [I,J] = find(~c.w{j});
%             tmp = x*c.L{j};
%             residual = c.A{j}(~c.w{j})-dotp(tmp,x,I,J);
%             f = f-norm(residual)^2;
%             df = df+2*sparse(I,J,residual,c.n(1),c.n(1))*tmp;
%         end
%     end
% end
f = 0.5*(f+c.r*norm(x,'fro')^2);
df = df+c.r*x;

end % evalP function
