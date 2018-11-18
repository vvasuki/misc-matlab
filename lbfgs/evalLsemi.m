function [f,df] = evalL(x,c)
% x: current Lambda matrix (f by f)
% c: other constants used to evaluate f and df
%    c.A: (cell) m adjacency matrices, each n by n dimension
%    c.w: (cell) 0/1 weight matrix
%    c.r: (scalar) regularization parameter
%    c.i: (scalar) index of current Lambda matrix
%    c.n: (matrix) size of the P matrix, dimension 2 by 1
%    c.P: (matrix) the P matrix, n by f dimension
%    c.L: (cell) m Lambda matrices, each f by f dimension

if nargin < 2
    error('Function evalL needs two input arguments!\n');
end

% calculate f and df
f = 0;
c.L{c.i} = x;

for j = 1:length(c.A)
    if nnz(c.w{j})/numel(c.w{j})==1
        f = f+norm(c.A{j},'fro')^2;
        if j == c.i
            tmp = x*c.P';
        else
            tmp = c.L{j}*c.P';
        end
        f = f-2*sum(dot(tmp',c.A{j}*c.P));
        tmp = tmp*c.P;
        if j == c.i
            df = -c.P'*(c.A{j}*c.P)+(c.P'*c.P)*tmp+c.r*x;
        end
        f = f+sum(dot(tmp',tmp))+c.r*norm(c.L{j},'fro')^2;
    else
        [I,J] = find(c.w{j});
        if j == c.i
            tmp = c.P*x;
        else
            tmp = c.P*c.L{j};
        end
        tmp = c.A{j}(c.w{j})-dotp(tmp,c.P,I,J);
        f = f+norm(tmp)^2;
        f = f+c.r*norm(c.L{j},'fro')^2;
        if j == c.i
            df = -c.P'*sparse(I,J,tmp,c.n(1),c.n(1))*c.P+c.r*x;
        end
    end
end
% for j = 1:length(c.A)
%     if nnz(c.w{j})/numel(c.w{j})<=0.5  % when W is very sparse
%         [I,J] = find(c.w{j});
%         if j == c.i
%             tmp = c.P*x;
%         else
%             tmp = c.P*c.L{j};
%         end
%         tmp = c.A{j}(c.w{j})-dotp(tmp,c.P,I,J);
%         f = f+norm(tmp)^2;
%         f = f+c.r*norm(c.L{j},'fro')^2;
%         if j == c.i
%             df = -c.P'*sparse(I,J,tmp,c.n(1),c.n(1))*c.P+c.r*x;
%         end
%     else % when W is very dense
%         f = f+norm(c.A{j},'fro')^2;
%         if j == c.i
%             tmp = x*c.P';
%         else
%             tmp = c.L{j}*c.P';
%         end
%         f = f-2*sum(dot(tmp',c.A{j}*c.P));
%         tmp = tmp*c.P;
%         if j == c.i
%             df = -c.P'*(c.A{j}*c.P)+(c.P'*c.P)*tmp+c.r*x;
%         end
%         f = f+sum(dot(tmp',tmp))+c.r*norm(c.L{j},'fro')^2;
%         if nnz(c.w{j})/numel(c.w{j})<1 % W is not totally dense
%             [I,J] = find(~c.w{j});
%             if j == c.i
%                 tmp = c.P*x;
%             else
%                 tmp = c.P*c.L{j};
%             end
%             tmp = c.A{j}(~c.w{j})-dotp(tmp,c.P,I,J);
%             f = f - norm(tmp)^2;
%             if j == c.i
%                 df = df + c.P'*sparse(I,J,tmp,c.n(1),c.n(1))*c.P;
%             end
%         end
%     end
% end
tmp = c.S*c.P;
f = 0.5*(f+c.r*norm(c.P,'fro')^2+c.b*sum(dot(c.P,tmp)));
%f = 0.5*(f + c.r*norm(c.P,'fro')^2);
end % evalL function
