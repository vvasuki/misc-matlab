function [Q, R] = QR(A)

[m, n] = size(A);

% Modified gram schmidt
V=A;
for j=[1:n]
    R(j,j)=norm(V(:,j),2);
    Q(:,j)=V(:,j)/R(j,j);
    for i=[j+1:n]
        R(j,i) = Q(:,j)'*V(:,i);
        V(:,i)=V(:,i)-R(j,i)*Q(:,j);
    end
end

end