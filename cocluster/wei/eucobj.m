function objval = eucobj(Z, W, rci, cci, Zm, Zr, Zc, basis)

% [objval] = eucobj(Z, W, rci, cci, Zm, Zr, Zc, basis)
%
% Compute the objective using squared Euclidean distance.
% 

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: eucobj.m,v 1.4 2008/05/04 23:04:15 wtang Exp $

if nargin < 8
	error('Eight input arguments are required!');
end

[nr,nc] = size(Z);
nrc = length(unique(rci));
R = spconvert([(1:nr)' rci ones(nr,1)]);
C = spconvert([(1:nc)' cci ones(nc,1)]);
[I,J,z] = find(Z);
switch basis
	case 1
		Zg = full(spdotdiv(sum(R'*Z,2),sum(R'*W,2)));
		Zh = full(spdotdiv(sum(Z*C,1),sum(W*C,1)));
		clear Z R C;
		h = Zg(rci(I)) + Zh(cci(J))' - Zm;
	case 2
		Zgh = full(spdotdiv(R'*Z*C,R'*W*C));
		clear Z R C;
		h = Zgh(rci(I) + nrc*(cci(J)-1));
	case 3
		Zgh = full(spdotdiv(R'*Z*C,R'*W*C));
		Zg = full(spdotdiv(sum(R'*Z,2),sum(R'*W,2)));
		clear Z R C;
		h = Zgh(rci(I) + nrc*(cci(J)-1)) + Zr(I) - Zg(rci(I));
	case 4
		Zgh = full(spdotdiv(R'*Z*C,R'*W*C));
		Zh = full(spdotdiv(sum(Z*C,1),sum(W*C,1)));
		clear Z R C;
		h = Zgh(rci(I) + nrc*(cci(J)-1)) + Zc(J)' - Zh(cci(J))';
	case 5
		Zgh = full(spdotdiv(R'*Z*C,R'*W*C));
		Zg = full(spdotdiv(sum(R'*Z,2),sum(R'*W,2)));
		Zh = full(spdotdiv(sum(Z*C,1),sum(W*C,1)));
		clear Z R C;
		h = Zgh(rci(I) + nrc*(cci(J)-1)) ...
			+ Zr(I) - Zg(rci(I)) + Zc(J)' - Zh(cci(J))';
	case 6
		Zgh = full(spdotdiv(R'*Z*C,R'*W*C));
		Zrh = full(spdotdiv(Z*C,W*C));
		Zgc = full(spdotdiv(R'*Z,R'*W));
		clear Z R C;
		h = Zrh(I+nr*(cci(J)-1)) + Zgc(rci(I) + nrc*(J-1)) ...
			- Zgh(rci(I) + nrc*(cci(J)-1));
	otherwise
		error('Basis should be within the range 1 - 6!');
end

objval = sum(nonzeros(W).*(z-h).^2);
end % main function