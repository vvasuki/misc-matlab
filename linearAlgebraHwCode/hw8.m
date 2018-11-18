A = [0 1
1 0]
[S L] = eig(A)
[Q U] = schur(A)

A = [0 0
1 0]
[S L] = eig(A)
[Q U] = schur(A)

A = [5 -3
-1 3]
[S L] = eig(A)
[Q U] = schur(A)

% Code for tridiagonal matrix eigenvector problem
m=5;
rand('seed',1);
A=randomTridiagonal(m);
% A

inverter = zeros(m);
for i=1:m
    inverter(i,m-i+1)=1;
end
A;
B=inverter*A*inverter;

[S L] = eig(A);
l_min = min(max(abs(L)));
i_min = find(max(abs(L)) == l_min,1);
l_min = L(i_min,i_min)
v_l_min = S(:,i_min)

x = evTridiagonalFromBS(A,l_min)
abs(x'*v_l_min/(norm(x)*norm(v_l_min)))
% norm(A*x-l_min*x)

x = inverter*evTridiagonalFromBS(B,l_min)
abs(x'*v_l_min/(norm(x)*norm(v_l_min)))
% norm(A*x-l_min*x)

m=100;
A = randomTridiagonal(m);
for i=1:m-1
    A(i,i+1) = A(i+1,i);
end
[S L] = eig(A);
absS = abs(S);
size(find(absS<=10^(-10)),1)
size(find(absS>10^(-10)),1)
y = zeros(10*m,1);
for i=1:10
    y((i-1)*m+1:i*m,1) = absS(:,i);
end
semilogy(y)

for i=1:m-1
    A(i+1,i) = 1;
    A(i,i+1) = 1;
    A(i,i) = -2;
end
A(m,m) = -2;
[S L] = eig(A);
absS = abs(S);
size(find(absS<=10^(-10)),1)
size(find(absS>10^(-10)),1)


