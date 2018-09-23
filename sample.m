% Reference: The SIU tutorial, internet.
% There are two kinds of m-files: the script
% files and the function files. Script files do not take the input arguments or return the output
% arguments. The function files may take input arguments or return output arguments.

% invoke this file by typing its name

function demonstrate
% dataTypes
% arithmatic
% algebra
% draw3dGraph
end

function getHelp
% help command Name
help rand
end

function dataTypes
% Create a row vector.
a = [1 2 3]

A=zeros(3,3);
A = [2 1;
     1 8]

% 2nd column of A
 A(:,2)
% 2nd row of A
 A(2, :)
 
str = 'programming in MATLAB is fun';

    m=800;
    A1 = rand(m);
    B1_0  = rand(size(A1));
    
    
    save data.mat A1;
    load data.mat;
end

function displayData
%  Greater display accuracy.
format long
% ; suppresses display of the content of vectors to the screen.
x = pi/100:pi/100:10*pi;
% Array x holds 1000 evenly spaced numbers
y = sin(x)./x;
% componentwise division of two arrays sin(x) and x. . before any operator
% is used for component-wise operation.
pause;
end

function algebra
symadd('x^2 + 2*x + 3','3*x+5')
end

function arithmatic
    A = rand(2);
    B = rand(2,3);
    C = A*B
    a = rand(2,1);
    b = rand(2,1);
    c = a'*b
    c = a.*b
end

function out = solveAxb(A, b)
% Solve Ax = b, perhaps using least squares
    x = A\b;
    out = x;

end

function drawGraph
% Drawing graphs
x=[10 30 55 80 100]
y=[.000217 0.001610 0.006002 0.014767 0.026210]
z=[0.003206 0.006004 0.014710 0.023637 0.035490]
plot(x,y,'r',x,z,'g')
xlabel('10 \leq rank(A) \leq 100')
ylabel('Runtime')
title('Plot of SVD program runtime against dimension of square matrix A')
h = legend('Matlab svd command','iterative svd procedure',2);
set(h,'Interpreter','none')
end

function draw3dGraph
% For z=f(x,y)
%  Make x and y 2 dimensional.
[x y] = meshgrid ( -14:0.2:14, -14:0.2:14 );
% z = x.^2 + y.^2-1
z = x.*y
axis equal;
axis square;
surf(x,y,z)
% mesh(x,y,z)
% contour ( x, y, z, [0 0])
grid
end


function inlineFunctions
% define a function that will be used during the current session only.
f = inline('sqrt(x.^2+y.^2)','x','y')
f(3,4)
end

function measureTime
% Measure time.
tic
q = pi;
while q > 0.01
    q = q/2;
end
toc
end

function controlStructures
    t0 = 1;
    t1 = [1 0];
    n=0
    if n == 0
       T = t0;
    elseif n == 1;
       T = t1;
    else
       for k=2:n
          T = [2*t1 0] - [0 0 t0];
          t0 = t1;
          t1 = T;
       end
    end

    x = ceil(10*rand); % Generate a random integer in {1, 2, ... , 10}
    switch x
    case {1,2}
    disp('Probability = 20%');
    case {3,4,5}
    disp('Probability = 30%');
    otherwise
    disp('Probability = 50%');
    end
end

function norms
%  Norms
norm(A,1)
norm(A,2)
norm(A,inf)
norm(A,'fro')
end

