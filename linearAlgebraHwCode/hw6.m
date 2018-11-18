m = 50;
x = 1;
W = eye(m);
for i = 1:m
    W(i,m) = 1;
for j = 1:i-1
    W(i,j) = -1;
end
end
W(m, m) = W(m, m) + x;
rcond(W);

I = eye(m);
y = W\I(:,1);

% W
A = [1 0;
    1 -3];
% 
% W = randn(m);

[L U P] =gepp(W);
norm(P*W - L*U)

% [L U P Q] =gecp(A)
% norm(P*A*Q - L*U)

[L U P Q] =gecp(W);
norm(P*W*Q - L*U)

% [L U P] =gepp(A)
% [L U P Q] =gecp(A)
