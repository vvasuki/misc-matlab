function result = rmse(train, probe, rci, cci, basis)

% [result] = rmse(train, probe, rci, cci, basis)
% Compute the RMSE based on the result from Bregman co-clustering.
%

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: rmse.m,v 1.3 2008/05/04 00:47:23 wtang Exp $

if nargin < 5
	error('Five input arguments are required!');
end

nrc = length(unique(rci));

% normalized the data by removing the global mean
[nr,nc] = size(train);
[I,J,xx] = find(train);
train = sparse(I,J,xx-mean(xx),nr,nc);

weight = spones(train);

R = spconvert([(1:nr)' rci ones(nr,1)]);
C = spconvert([(1:nc)' cci ones(nc,1)]);

[I,J,rating] = find(probe);
clear probe;

Zm = full(sum(sum(train))/sum(sum(weight)));     % global mean
Zr = full(spdotdiv(sum(train,2),sum(weight,2))); % row mean
Zc = full(spdotdiv(sum(train,1),sum(weight,1))); % column mean

switch basis
	case 1
		Zg = full(spdotdiv(sum(R'*train,2),sum(R'*weight,2)));
		Zh = full(spdotdiv(sum(train*C,1),sum(weight*C,1)));
		predict = Zg(rci(I)) + Zh(cci(J))' - Zm;
	case 2
		Zgh = full(spdotdiv(R'*train*C,R'*weight*C));
		predict = Zgh(rci(I) + nrc*(cci(J)-1));
	case 3
		Zgh = full(spdotdiv(R'*train*C,R'*weight*C));
		Zg = full(spdotdiv(sum(R'*train,2),sum(R'*weight,2)));
		predict = Zgh(rci(I) + nrc*(cci(J)-1)) + Zr(I) - Zg(rci(I));
	case 4
		Zgh = full(spdotdiv(R'*train*C,R'*weight*C));
		Zh = full(spdotdiv(sum(train*C,1),sum(weight*C,1)));
		predict = Zgh(rci(I) + nrc*(cci(J)-1)) + Zc(J)' - Zh(cci(J))';
	case 5
		Zgh = full(spdotdiv(R'*train*C,R'*weight*C));
		Zg = full(spdotdiv(sum(R'*train,2),sum(R'*weight,2)));
		Zh = full(spdotdiv(sum(train*C,1),sum(weight*C,1)));
		predict = Zgh(rci(I) + nrc*(cci(J)-1)) ...
			+ Zr(I) - Zg(rci(I)) + Zc(J)' - Zh(cci(J))';
	case 6
		Zgh = full(spdotdiv(R'*train*C,R'*weight*C));
		Zrh = full(spdotdiv(train*C,weight*C));
		Zgc = full(spdotdiv(R'*train,R'*weight));
		predict = Zrh(I+nr*(cci(J)-1)) + Zgc(rci(I) + nrc*(J-1)) ...
			- Zgh(rci(I) + nrc*(cci(J)-1));
	otherwise
		error('Basis should be within the range 1 - 6!');
end

result = sqrt(sum((predict-rating+mean(xx)).^2)/length(rating));
end % main function