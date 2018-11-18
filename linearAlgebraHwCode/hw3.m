% stuff for question 1.
A = [-2 11
    -10 5] 
% A = rand(2,2).*10
[U,S,V]=svd(A)
A*A'
[X_AAt,L_AAt] = eig(A'*A)


US=U*S
norm(A,1)
norm(A,2)
norm(A,inf)
norm(A,'fro')
inv(A)
V*inv(S)*U'
[X,L]=eig(A)
det(A)
pi*S(1,1).*S(2,2)


% Draw oval and ellipse
% f(x,y) = x^2+y^2
[x y] = meshgrid ( -14:0.2:14, -14:0.2:14 );
z = x.^2 + y.^2 - 1;
theta = atan(U(1,1)/U(1,2))*ones(size(x));
w = (1/S(1,1)^2)*((x.^2+y.^2).^(1/2).*cos(atan(y./x) - theta)).^2 + (1/S(2,2)^2)*((x.^2+y.^2).^(1/2).*sin(atan(y./x) - theta)).^2 - 1;
u1 = y-x*(U(1,1)/U(1,2));
u2 = y-x*(U(2,1)/U(2,2));
axis equal;
axis square;
contour ( x, y, z, [0 0])
hold
contour ( x, y, w, [0 0])
hold
grid
hold
% contour ( x, y, u2, [0 0] )
% hold

% stuff for question 2.
T=[zeros(2) A'
    A zeros(2)]
U
S
V
[Xt,Lt]=eig(T)

eps

