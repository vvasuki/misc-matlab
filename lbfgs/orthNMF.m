function [P,L] = orthNMF(A,k)

m = length(A);
n = size(A{m},1);

% initialize P and Lambda_i
P = rand(n,k);
%B = A{1};
%for i = 2:m
%    B = B+A{i};
%end
%[P,v] = eigs(B/m,k);
for i = 1:m
    L{i} = rand(k);
    L{i} = (L{i}+L{i}')/2;
end

% parameters
maxi = 1000;    % maximum number of iterations
relerr = 1e-6; % relative error criteria
eps = 1e-9;    % small value

for i = 1:maxi
    % updating P
    tmp1 = zeros(size(P));
    tmp2 = zeros(size(P));
    for j = 1:m
        tmp1 = tmp1+(A{j}*P*L{j});
        tmp2 = tmp2+(P*((P'*A{j}*P)*L{j}));
    end
    P = max(P.*(tmp1./tmp2),eps);

    % updating Lambda_i
    for j = 1:m
        L{j} = L{j}.*(P'*A{j}*P)./((P'*P)*L{j}*(P'*P));
        L{j} = max(L{j},eps);
    end

    % compute the objective
    if i > 1
        old_obj = obj;
    end
    obj = 0;
    for j = 1:m
        obj = obj+0.5*norm(A{j}-P*L{j}*P','fro')^2;
    end
    fprintf('iter %d: obj %.4f\n',i,obj);

    if i > 1
        if old_obj-obj<old_obj*relerr
            break;
        end
    end
end

end % orthNMF function
