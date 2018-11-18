classdef R2Geometry
methods(Static=true)
function area = ellipseArea_Mc(M)
%      Area of the ellipse: { Mx + c| \norm{x} \leq 1}
    [U S V] = svd(M);
    area = pi* prod(sqrt(diag(S)));
end

function area = ellipseArea_Ab(A)
%      Finds loci of the ellipse: $\set{x | \norm{M^{-1}x - M^{-1}c} = 1} = \set{x: \norm{Ax + b } = 1}$
    import topology.*;
    area = R2Geometry.ellipseArea_Mc(pinv(A));
end

function ellipse = ellipseLoci_Mc(M, c)
%      Finds loci of the ellipse: { Mx + c| \norm{x} \leq 1}
    ang = linspace(0,2*pi,201);
    numLoci = numel(ang);
    X = [ cos(ang) ; sin(ang) ];
%      keyboard
    ellipse = M*X + diag(c)*ones(2, numLoci);
end

function ellipse = ellipseLoci_Ab(A, b)
%      Finds loci of the ellipse: $\set{x | \norm{M^{-1}x - M^{-1}c} = 1} = \set{x: \norm{Ax + b } = 1}$
    import topology.*;
    M = pinv(A);
    c = -M*b;
    ellipse = R2Geometry.ellipseLoci_Mc(M, c);
end

function ellipse = ellipseLoci_Pqr(P, q, r)
%      Finds loci of the ellipse: { x | (1/2) x^T P x + q^T x + r <= 0 }
%      Credits: Dr. Michael Grant.
    A = sqrtm( P );
    b  = A \ q;
    w  = sqrt( b' * b - 2 * r );
    A  = A / w;
    b  = b / w;
    ang = linspace(0,2*pi,201);
    cs = [ cos(ang) ; sin(ang) ];
    ellipse = A \ ( cs - b(:,ones(1,201)) );
end

function testEllipseLoci
    import topology.*;
    M = [2 0;0 1];
    c = [4; 3];
    ellipse = R2Geometry.ellipseLoci_Mc(M, c);
    plot(ellipse(1,:), ellipse(2,:));
    area = R2Geometry.ellipseArea_Mc(M)
    fprintf('Expected: %d\n', pi*prod(sqrt(diag(M))));
end

function testClass
    display 'Class definition is ok';
end

end
end
