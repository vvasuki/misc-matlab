function x = evTridiagonalFromBS(A, l_min)

m = size(A,1);
AlI= A - l_min*eye(m);
x= zeros(m,1);
b= zeros(m-1,1);
x(1,1) = 1;
b(1,1) = (0 - x(1,1)*AlI(1,1));
b(2,1) = (0 - x(1,1)*AlI(2,1));
x(2:m,1) = AlI(1:m-1,2:m)\b;
end