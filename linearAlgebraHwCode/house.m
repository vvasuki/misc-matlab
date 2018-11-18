% Householder triangularization.
% Gives:
% R upper triangular matrix
% W $m\times n$, lower triangular, its columns are the vectors $v_{k}$ 
% defining the successive Householder reflections.

function [W, R] = house(A)
[m, n] = size(A);
R = A;
W = zeros(m,n);
for k = 1:n
    x = R(k:m,k);
    I = eye(m-k+1);
    e_1 = I(:,1);
    sgn = sign(x(1,1));
    if(sgn == 0)
        sgn = 1;
    end
    v = -sgn*norm(x).*e_1 - x;
    v = v./norm(v);
    R(k:m,k:n) = R(k:m,k:n) - 2.*v*v'*R(k:m,k:n);
    W(k:m,k) = W(k:m,k) + v;
end

end