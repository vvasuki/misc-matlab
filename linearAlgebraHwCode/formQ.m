% Q = formQ(W) that takes the matrix W produced by house as input and
% generates a corresponding $m \times m$ orthogonal matrix Q.

function Q = formQ(W)
[m, n] = size(W);
Q = eye(m,m);
for k = 1:n
    Q_k = eye(m,m);
    v = W(k:m,k);
    F = eye(m-k+1,m-k+1) - 2*v*v';
    Q_k(k:m,k:m) = F;
    Q = Q*Q_k;
end

end