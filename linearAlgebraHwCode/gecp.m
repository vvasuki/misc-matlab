% Gaussian elimination with complete pivoting.
% Needs square A
% Gives:
% L, U and P

function [L, U, P, Q] = gecp(A)
[m, m] = size(A);
U = A;
L = eye(m);
P = eye(m);
Q = eye(m);
for k=1:m-1
    [maxRowIndeces, maxColIndeces] = find(abs(U(k:m, k:m)) == max(max(abs(U(k:m, k:m)))));
    maxRowIndex = k-1 + maxRowIndeces(1,1);
    maxColIndex = k-1 + maxColIndeces(1,1);
    
%  Exchange rows
    tmp = U(k, k:m);
    U(k, k:m) = U(maxRowIndex, k:m);
    U(maxRowIndex, k:m) = tmp;
    
    tmp = L(k, 1:k-1);
    L(k, 1:k-1) = L(maxRowIndex, 1:k-1);
    L(maxRowIndex, 1:k-1) = tmp;
    
    tmp = P(k, :);
    P(k, :) = P(maxRowIndex, :);
    P(maxRowIndex, :) = tmp;

%  Exchange cols
    tmp = U(:, k);
    U(:, k) = U(:, maxColIndex);
    U(:, maxColIndex) = tmp;
    
    
    tmp = Q(:, k);
    Q(:, k) = Q(:, maxColIndex);
    Q(:, maxColIndex) = tmp;

    for j=k+1:m
        L(j,k) = U(j,k)/U(k,k);
        U(j,k:m) = U(j,k:m) - L(j,k)*U(k,k:m);
    end

end
end
