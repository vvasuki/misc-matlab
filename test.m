A = [.3 .2
    .2 .3]

norm3 = inline ('nthroot((A(1,1)^3 + A(1,2)^3 + A(2,1)^3 + A(2,2)^3),3)', 'A')
norm3(A)
I = eye(2)
I-A
inv(I-A)
norm3(inv(I-A))
1/(1-norm3(A))