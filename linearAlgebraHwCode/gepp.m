% Gaussian elimination with partial pivoting.
% Needs square A
% Gives:
% L, U and P

function [L, U, P] = gepp(A)
[m, m] = size(A);
U = A;
L = eye(m);
P = eye(m);
for k=1:m-1
    maxI = k-1 + find(abs(U(k:m, k)) == max(abs(U(k:m, k))));
    i = maxI(1,1);
    
    tmp = U(k, k:m);
    U(k, k:m) = U(i, k:m);
    U(i, k:m) = tmp;
    
    tmp = L(k, 1:k-1);
    L(k, 1:k-1) = L(i, 1:k-1);
    L(i, 1:k-1) = tmp;
    
    tmp = P(k, :);
    P(k, :) = P(i, :);
    P(i, :) = tmp;

    for j=k+1:m
        L(j,k) = U(j,k)/U(k,k);
        U(j,k:m) = U(j,k:m) - L(j,k)*U(k,k:m);
    end

end
end
