function [f,df] = evalF(x,c)
% Evaluate the objective function of Linked Matrix Factorization.
%
% x: the P and Lambda matrices (n by (m+1)f)
% c: other constants used to evaluate f and df
%    c.A: (cell) m adjacency matrices, each in dimension n by n
%    c.r: (scalar) regularization parameter
%    c.n: (matrix) size of the P matrix, dimension 2 by 1
%    c.w: (cell) true if A is dense, otherwise 0/1 weight matrix

% By Wei Tang (wtang@cs.utexas.edu)

if nargin < 2
    error('Function evalD needs two input arguments!\n');
end

P = x(1:c.n(1),:);
L = mat2cell(x(c.n(1)+1:end,:),repmat(c.n(2),1,length(c.A)),c.n(2));

% calculate f and df
f = 0;                                  % objective value
df = zeros(size(x));                    % gradient of P and Lambda matrices

for j = 1:length(c.A)
    if nnz(c.w{j})/numel(c.w{j})<=0.5 % when W is very sparse
        [I,J] = find(c.w{j});
        tmp = P*L{j};
        residual = c.A{j}(c.w{j})-dotp(tmp,P,I,J);
        f = f+norm(residual,'fro')^2;
        f = f+c.r*norm(L{j},'fro')^2;
        residual = sparse(I,J,residual,c.n(1),c.n(1));
        df(1:c.n(1),:) = df(1:c.n(1),:)-2*residual*tmp;
        df(c.n(1)+(j-1)*c.n(2)+1:c.n(1)+j*c.n(2),:) = -P'*residual*P+c.r*L{j};
    else % when W is very dense
        f = f+norm(c.A{j},'fro')^2;
        tmp = P*L{j};
        f = f-2*sum(dot(tmp,c.A{j}*P));
        tmp1 =P'*tmp;
		df(1:c.n(1),:) = df(1:c.n(1),:)-2*c.A{j}*tmp+2*tmp*(tmp1);
        df(c.n(1)+(j-1)*c.n(2)+1:c.n(1)+j*c.n(2),:) = -P'*(c.A{j}*P)+tmp1*(P'*P)+c.r*L{j};
        f = f+sum(dot(tmp1',tmp1));
		f = f+c.r*norm(L{j},'fro')^2;
        if nnz(c.w{j})/numel(c.w{j})<1 % when W is not totally dense
            [I,J] = find(~c.w{j});
            tmp = P*L{j};
            residual = c.A{j}(~c.w{j})-dotp(tmp,P,I,J);
            f = f-norm(residual,'fro')^2;
            df(1:c.n(1),:) = df(1:c.n(1),:)+2*sparse(I,J,residual,c.n(1),c.n(1))*tmp;
            df(c.n(1)+(j-1)*c.n(2)+1:c.n(1)+j*c.n(2),:) = df(c.n(1)+(j-1)*c.n(2)+1:c.n(1)+j*c.n(2),:)...
                + c.P'*sparse(I,J,residual,c.n(1),c.n(1))*c.P;
        end
    end
end

% if isscalar(c.w)
% 	for j = 1:length(c.A)
% 		tmp = P*L{j};
% 		residual = c.A{j}-tmp*P';
% 		f = f+norm(residual,'fro')^2;
% 		f = f+c.r*norm(L{j},'fro')^2;
% 		df(1:c.n(1),:) = df(1:c.n(1),:)-2*residual*tmp;
% 		df(c.n(1)+(j-1)*c.n(2)+1:c.n(1)+j*c.n(2),:) = -P'*residual*P+c.r*L{j};
% 	end
% else
% 	for j = 1:length(c.A)
% 		[I,J] = find(c.w{j});
% 		tmp = P*L{j};
% 		residual = nonzeros(c.A{j}(logical(c.w{j})))-dotp(tmp,P,I,J);
% 		f = f+norm(residual)^2;
% 		f = f+c.r*norm(L{j},'fro')^2;
% 		residual = sparse(I,J,residual,c.n(1),c.n(1));
% 		df(1:c.n(1),:) = df(1:c.n(1),:)-2*residual*tmp;
% 		df(c.n(1)+(j-1)*c.n(2)+1:c.n(1)+j*c.n(2),:) = -P'*residual*P+c.r*L{j};
% 	end
% end
f = 0.5*(f+c.r*norm(P,'fro')^2);
df(1:c.n(1),:) = df(1:c.n(1),:)+c.r*P;

end % evalF function
