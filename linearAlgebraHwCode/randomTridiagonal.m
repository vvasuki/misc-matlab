function A = randomTridiagonal(m)

A = randn(m);
for i=1:m
    for j=1:m
        if(i < j-1 || i > j+1 || j < i-1 || j > i+1)
            A(i,j)=0;
        end

    end
end
end