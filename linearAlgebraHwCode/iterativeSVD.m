n = 80;
tol = 1e-5;

% Pick a non singular matrix A
A = fix(10*rand(n, n));
while det(A) == 0
    A = fix(10*rand(n, n));
end

% A = [2 1;
%     1,8]

% tic
[UMatlab, SMatlab, VMatlab] = svd(A);
% toc

tic
% Initialize $U,\ \Sigma,\ V$ to $I$.
U = eye(n);
V = eye(n);
S = eye(n);
svdError = 500;
svdErrorPrev =  svdError*1000;
iterations = 0;

while ((svdError > tol))
    % Assuming $U,\ \Sigma$ fixed, compute $V$ and orthogonalize it.
    % A=USV'
    % AV=US
    % V = inv(A)*U*S; Won't lead to convergence.
    % V = (inv(S)*U'*A)'; Gives different V, leads to convergence!
    [Q,R] = qr((inv(S)*U'*A)');
    V = Q;

    % Assume $U, V$ fixed, compute $\Sigma$. Ensure that $\Sigma$ is diagonal and positive.

    % Equivalent but shorter procedure
      % AV = US
    % Av_i = u_i s_i
    % I do the following for all i = 1:n
    % Project Av_i on u_i to get the magnitude (p_i) of the projection along the direction of u_i, solve p_i = norm(u_i) s_i. (I find p_i by calculating u_i' Av_i/ norm(u_i).)
    % Fix sign of u_i and s_i if s_i<0.
    % Update matreces V and S.
    % AV = US
    S_raw = U'*A*V;
    S = S_raw;
    S = S.*eye(n);
    for i=1:n
        if S(i,i)<0
            S(i,i) = -S(i,i);
            V(:,i) = -V(:,i);
        end
    end



    % Assuming $\Sigma,\ V$ fixed, compute $U$ and orthogonalize it.
    % AV = US
    U = A*V*inv(S);
    [Q,R] = qr(U);
    U = Q;

    % If $\|A-U\Sigma V^T\|_F\geq tol$, repeat steps (ii)-(iv).
    % Faulty calculation:
    % svdErrorMatrix = A- (U*S_raw*V');
    svdErrorMatrix = A- (U*S*V');
    svdError = norm(svdErrorMatrix,'fro');
    
    iterations = iterations + 1;
    
end
toc

% svdError
% iterations

% A
% UMatlab
% U
% SMatlab
% S
% VMatlab
% V

x=[10 30 55 80]
y=[.000217 0.001610 0.006002 0.014767]
z=[0.015000 0.372791 3.718036 11.570505]
plot(x,y,'r',x,z,'g')
xlabel('10 \leq rank(A) \leq 100')
ylabel('Runtime')
title('Plot of SVD program runtime against dimension of square matrix A')
h = legend('Matlab svd command','iterative svd procedure',2);
set(h,'Interpreter','none')
grid
