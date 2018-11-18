m = 50;
n = 12;

t = linspace(0,1,m);
t=t';
b = cos(4.*t);
% Use vander and fliplr to define A to be matrix associated with least
% squares fitting on this grid by polynomial of degree n-1.
A = fliplr(vander(t));
A = A(:,1:n);

format long;

R=chol(A'*A);
w=R'\(A'*b);
x=R\w

[Q, R] = mgs(A);
x = (R\Q'*b)

[W, R] = house(A);
Q = formQ(W);
x = (R\(Q'*b))

[Q, R] = qr(A);
x = (R\(Q'*b))

x=A\b

[U S V] = svd(A);
w=(S\(U'*b));
x=V*w

%  syms t   % t is a symbolic variable
%  F = [-cos(t) sin(t); sin(t) cos(t)]
%  x = zeros(2)\(F+eye(2))
%  best = simple(ans)
%  

