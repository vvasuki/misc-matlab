e= sqrt(10^-16)
format long
1+e^2

A=[ 1 1 1
    e 0 0
    0 e 0
    0 0 e]
n=3;

% Classical gram schmidt
R=zeros(3,3);
Q=A;

for j=[1:n]
    v = A(:,j);
    for i=[1:j-1]
        R(i,j) = Q(:,i)'*A(:,j);
        v=v-R(i,j)*Q(:,i);
    end
    R(j,j)=norm(v,2);
    Q(:,j)=v/R(j,j);
end

Q
for j=[1:n]
    for i=[1:j-1]
        Q(:,i)'*Q(:,j)
    end
end


% Modified gram schmidt
V=A
for j=[1:n]
    R(j,j)=norm(V(:,j),2);
    Q(:,j)=V(:,j)/R(j,j);
    for i=[j+1:n]
        R(j,i) = Q(:,j)'*V(:,i);
        V(:,i)=V(:,i)-R(j,i)*Q(:,j);
    end
end

Q
for j=[1:n]
    for i=[1:j-1]
        Q(:,i)'*Q(:,j)
    end
end
