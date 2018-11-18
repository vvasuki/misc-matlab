function [rci, cci, minobj, rec] = icc(mat, nrc, ncc, varargin)

% ICC Bregman co-clustering algorithm using I divergence.
%
% [rci, cci, minobj] = icc(mat, nrc, ncc, [[argID,] value, ...])
%
% Examples:
%	[rci, cci] = icc(mat, nrc, ncc)
%	[rci, cci, minobj] = ecc(mat, nrc, ncc)
%
% INPUT:
%	mat     (matrix) 2-D matrix with any real values, size nr by nc
%	nrc     (scalar) number of row clusters
%	ncc     (scalar) number of column clusters
%
%	Valid argument IDs and corresponding values are:
%	  'maxi'      (scalar) maximum number of iterations (default = 50)
%	  'relerr'    (scalar) relative stopping criterion (default = 1e-5)
%	  'nreps'     (scalar) number of repeats (default = 1)
%	  'basis'     (scalar) basis of bregman co-clustering (default = 5)
%     'precision' (scalar) matrix precision (default = 1e-9)
%	  'weight'    (matrix) the weight matrix, same size of mat
%
% OUTPUT:
%	rci    (vector) resulting row cluster identities, length nr
%	cci    (vector) resulting column cluster identities, length nc
%   minobj (scalar) the minimum objective value in multiple runs
%   rec    (structor) the record of each run
%
% References:
% A Generalized Maximum Entropy Approach to Bregman Co-Clustering and
% Matrix Approximations. A. Banerjee, I. S. Dhillon, J. Ghosh, S. Merugu,
% and D. S. Modha. Journal of Machine Learning Research (JMLR), vol. 8,
% pages 1919-1986, August 2007.
%
% See also: ecc

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: icc.m,v 1.7 2008/05/02 21:03:33 wtang Exp $

% check input arguments
error(nargchk(3, 19, nargin));

% make sure the ncc is always no less than nrc
transposed = nrc > ncc;
if transposed
	mat = mat';
	[nrc, ncc] = deal(ncc, nrc);
end

% size of the input matrix
[nr, nc] = size(mat);
if nrc > nr || nrc < 1 
	error('The number of row clusters must be between 1 and nr!');
end
if ncc > nc || ncc < 1
	error('The number of column clusters must be between 1 and nc!');
end

% default input arguments
maxi = 50;
relerr = 1e-5;
nreps = 1;
basis = 5;
prec = 1e-9;
wgt = spones(mat);
rci = [];
cci = [];

% parse varargin
i = 1;
while i <= length(varargin)
	if ischar(varargin{i})
		switch varargin{i}
			case 'maxi'
				i = i+1; maxi = varargin{i};
			case 'relerr'
				i = i+1; relerr = varargin{i};
			case 'nreps'
				i = i+1; nreps = varargin{i};
			case 'basis'
				i = i+1; basis = varargin{i};
			case 'precision'
				i = i+1; prec = varargin{i};
			case 'weight'
				i = i+1; wgt = varargin{i};
			case 'rci'
				i = i+1; rci = varargin{i};
			case 'cci'
				i = i+1; cci = varargin{i};
			otherwise
				error('Ignoring invalid argID %s', varargin{i});
		end
	else
		error('Ignoring invalid argument #%d', i+2);
	end
	i = i+1;
end

% randomly initialize the row cluster indicators
rand('twister',sum(100*clock));
if isempty(rci)
	rci = mod((1:nr)', nrc)+1;
	rci = rci(randperm(nr));
end

% randomly initialize the column cluster indicators
if isempty(cci)
	cci = mod((1:nc)', ncc)+1;
	cci = cci(randperm(nc));
end

% check the input arguments
if any(any(mat<0))
	error('All elements in the matrix should be positive');
end
if size(wgt) ~= size(mat)
	error('The weight matrix should be in the same size of data matrix');
end
if basis < 1 || basis > 6
	error('The basis should be in the range of 1 to 6');
end
if length(rci) ~= nr
	error('The length of rci must be %d!', nd);
end
if length(cci) ~= nc
	error('The length of cci must be %d!', nw);
end
if min(rci) < 1 || max(rci) > nrc
	error('The values in rci must be between 1 and %d!', nrc);
end
if min(cci) < 1 || max(cci) > ncc
	error('The values in cci must be between 1 and %d!', ncc);
end

% pre-compute some constants
if basis == 6
	orig_mat = mat;
end
mat = wgt.*mat;
Zm = full(sum(sum(mat))/sum(sum(wgt)));     % global mean
Zr = full(spdotdiv(sum(mat,2),sum(wgt,2))); % row mean
Zc = full(spdotdiv(sum(mat,1),sum(wgt,1))); % column mean

% the initial row/column indicator matrix
init_rci = rci;
init_cci = cci;

rec.obj = cell(1, nreps); % the trial of objectives in each run
rec.rci = cell(1, nreps); % the rci in the corresponding run
rec.cci = cell(1, nreps); % the cci in the corresponding run
obj = zeros(1, nreps);    % the minimum objective in each run

% repeat co-clustering nreps runs
for t = 1:nreps
	fprintf('trial: %d\n',t);
	rci = init_rci; cci = init_cci;
	R = spconvert([(1:nr)' rci ones(nr,1)]);
	C = spconvert([(1:nc)' cci ones(nc,1)]);
	rec.obj{t} = zeros(1, maxi);
    
	switch basis
		case 1
			for i = 1:maxi
				% basis 1 row update
				ZrowRed = spdotdiv(mat*C,wgt*C);
				Zh = full(spdotdiv(sum(mat*C,1),sum(wgt*C,1)));
				ZrowCRed = spdotdiv(wgt./Zm*diag(sparse(Zh*C'))*C,wgt*C);
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZrowVRed = Zg(:,ones(1,ncc));
				ZrowVRed(logical(ZrowVRed==0)) = prec;
				% compute row assignement distance matrix
				D = idivergence(ZrowRed, ZrowCRed, ZrowVRed, wgt*C);
				% update the row cluster indicator
				[minvals, rci] = min(D, [], 2);
				R = spconvert([(1:nr)' rci ones(nr,1)]);

				% basis 1 column update
				ZcolRed  = spdotdiv(R'*mat,R'*wgt);
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZcolCRed = spdotdiv(R'*diag(sparse(R*Zg))*wgt./Zm,R'*wgt);
				ZcolVRed = Zh(ones(1,nrc),:);
				ZcolVRed(logical(ZcolVRed==0)) = prec;
				% compute column assignment distance matrix
				D = idivergence(ZcolRed', ZcolCRed', ZcolVRed', wgt'*R);
				% update the column cluster indicator
				[minvals, cci] = min(D, [], 2);
				C = spconvert([(1:nc)' cci ones(nc,1)]);

				% current objective value
				obj = idivobj(mat, wgt, rci, cci, Zm, Zr, Zc, 1);
				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				rec.obj{t}(i) = obj;
				fprintf('iter %d: obj %f\n',i,obj);
			end
		case 2
			for i = 1:maxi
				% basis 2 row update
				ZrowRed = spdotdiv(mat*C,wgt*C);
				ZrowCRed = spones(wgt*C);
				ZrowVRed = full(spdotdiv(R'*mat*C,R'*wgt*C));
				ZrowVRed(logical(ZrowVRed==0)) = prec;
				% compute row assignement distance matrix
				D = idivergence(ZrowRed, ZrowCRed, ZrowVRed, wgt*C);
				% update the row cluster indicator
				[minvals, rci] = min(D, [], 2);
				R = spconvert([(1:nr)' rci ones(nr,1)]);

				% basis 2 column update
				ZcolRed = spdotdiv(R'*mat,R'*wgt);
				ZcolCRed = spones(R'*wgt);
				ZcolVRed = full(spdotdiv(R'*mat*C,R'*wgt*C));
				ZcolVRed(logical(ZcolVRed==0)) = prec;
				% compute column assignment distance matrix
				D = idivergence(ZcolRed', ZcolCRed', ZcolVRed', wgt'*R);
				% update the column cluster indicator
				[minvals, cci] = min(D, [], 2);
				C = spconvert([(1:nc)' cci ones(nc,1)]);

				% current objective value
				obj = idivobj(mat, wgt, rci, cci, Zm, Zr, Zc, 2);
				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				rec.obj{t}(i) = obj;
				fprintf('iter %2d: obj %f\n',i,obj);
			end
		case 3
			for i = 1:maxi
				% basis 3 row update
				ZrowRed = spdotdiv(mat*C,wgt*C);
				ZrowCRed = spdotdiv(diag(sparse(Zr))*wgt*C,wgt*C);
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZrowVRed = full(diag(sparse(1./Zg))*spdotdiv(R'*mat*C,R'*wgt*C));
				ZrowVRed(logical(ZrowVRed==0)) = prec;
				% compute row assignement distance matrix
				D = idivergence(ZrowRed, ZrowCRed, ZrowVRed, wgt*C);
				% update the row cluster indicator
				[minvals, rci] = min(D, [], 2);
				R = spconvert([(1:nr)' rci ones(nr,1)]);

				% basis 3 column update
				ZcolRed = spdotdiv(R'*mat,R'*wgt);
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZcolCRed = spdotdiv(R'*diag(sparse(Zr./sparse(R*Zg)))*wgt,R'*wgt);
				ZcolVRed = full(spdotdiv(R'*mat*C,R'*wgt*C));
				ZcolVRed(logical(ZcolVRed==0)) = prec;
				% compute column assignment distance matrix
				D = idivergence(ZcolRed', ZcolCRed', ZcolVRed', wgt'*R);
				% update the column cluster indicator
				[minvals, cci] = min(D, [], 2);
				C = spconvert([(1:nc)' cci ones(nc,1)]);

				% current objective value
				obj = idivobj(mat, wgt, rci, cci, Zm, Zr, Zc, 3);
				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				rec.obj{t}(i) = obj;
				fprintf('iter %d: obj %f\n',i,obj);
			end
		case 4
			for i = 1:maxi
				% basis 4 row update
				ZrowRed = spdotdiv(mat*C,wgt*C);
				Zh = full(spdotdiv(sum(mat*C,1),sum(wgt*C,1)));
				ZrowCRed = spdotdiv(wgt*diag(sparse(Zc)./sparse(Zh*C'))*C,wgt*C);
				ZrowVRed = full(spdotdiv(R'*mat*C,R'*wgt*C));
				ZrowVRed(logical(ZrowVRed==0)) = prec;
				% compute row assignement distance matrix
				D = idivergence(ZrowRed, ZrowCRed, ZrowVRed, wgt*C);
				% update the row cluster indicator
				[minvals, rci] = min(D, [], 2);
				R = spconvert([(1:nr)' rci ones(nr,1)]);

				% basis 4 column update
				ZcolRed = spdotdiv(R'*mat,R'*wgt);
				ZcolCRed = spdotdiv(R'*wgt*diag(sparse(Zc)),R'*wgt);
				Zh = full(spdotdiv(sum(mat*C,1),sum(wgt*C,1)));
				ZcolVRed = full(spdotdiv(R'*mat*C,R'*wgt*C)*diag(sparse(1./Zh)));
				ZcolVRed(logical(ZcolVRed==0)) = prec;
				% compute column assignment distance matrix
				D = idivergence(ZcolRed', ZcolCRed', ZcolVRed', wgt'*R);
				% update the column cluster indicator
				[minvals, cci] = min(D, [], 2);
				C = spconvert([(1:nc)' cci ones(nc,1)]);

				% current objective value
				obj = idivobj(mat, wgt, rci, cci, Zm, Zr, Zc, 4);

				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				rec.obj{t}(i) = obj;
				fprintf('iter %d: obj %f\n',i,obj);
			end
		case 5
			for i = 1:maxi
				% basis 5 row update
				ZrowRed = spdotdiv(mat*C,wgt*C);
				Zh = full(spdotdiv(sum(mat*C,1),sum(wgt*C,1)));
				ZrowCRed = spdotdiv(diag(sparse(Zr)) * wgt ...
					* diag(sparse(Zc./(Zh*C')))*C,wgt*C);
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZrowVRed = full(diag(sparse(1./Zg))*spdotdiv(R'*mat*C,R'*wgt*C));
				ZrowVRed(logical(ZrowVRed==0)) = prec;
				% compute row assignement distance matrix
				D = idivergence(ZrowRed, ZrowCRed, ZrowVRed, wgt*C);
				% update the row cluster indicator
				[minvals, rci] = min(D, [], 2);
				R = spconvert([(1:nr)' rci ones(nr,1)]);

				% basis 5 column update
				ZcolRed = spdotdiv(R'*mat,R'*wgt);
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZcolCRed = spdotdiv(R'*diag(sparse(Zr./(R*Zg))) * wgt ...
					* diag(sparse(Zc)),R'*wgt);
				Zh = full(spdotdiv(sum(mat*C,1),sum(wgt*C,1)));
				ZcolVRed = full(spdotdiv(R'*mat*C,R'*wgt*C)*diag(sparse(1./Zh)));
				ZcolVRed(logical(ZcolVRed==0)) = prec;
				% compute column assignment distance matrix
				D = idivergence(ZcolRed', ZcolCRed', ZcolVRed', wgt'*R);
				% update the column cluster indicator
				[minvals, cci] = min(D, [], 2);
				C = spconvert([(1:nc)' cci ones(nc,1)]);

				% current objective value
				obj = idivobj(mat, wgt, rci, cci, Zm, Zr, Zc, 5);
				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				rec.obj{t}(i) = obj;
				fprintf('iter %d: obj %f\n',i,obj);
			end
		case 6
			for i = 1:maxi
				% basis 6 row update
				ZrowConst = spdotdiv(mat*C,wgt*C);
				ZrowVar = full(spdotdiv(spdotdiv(R'*mat,R'*wgt), ...
					spdotdiv(R'*mat*C,R'*wgt*C)*C'));
				ZrowVar(logical(ZrowVar==0)) = prec;
				% compute row assignement distance matrix
				D = idivergence6(orig_mat, ZrowConst, cci, ZrowVar, wgt);
				% update the row cluster indicator
				[minvals, rci] = min(D, [], 2);
				R = spconvert([(1:nr)' rci ones(nr,1)]);

				% basis 6 column update
				ZcolConst = spdotdiv(R'*mat,R'*wgt);
				ZcolVar = full(spdotdiv(spdotdiv(mat*C,wgt*C), ...
					R*spdotdiv(R'*mat*C,R'*wgt*C)));
				ZcolVar(logical(ZcolVar==0)) = prec;
				% compute column assignment distance matrix
				D = idivergence6(orig_mat', ZcolConst', rci, ZcolVar', wgt');
				% update the column cluster indicator
				[minvals, cci] = min(D, [], 2);
				C = spconvert([(1:nc)' cci ones(nc,1)]);

				% current objective value
				obj = idivobj(mat, wgt, rci, cci, Zm, Zr, Zc, 6);
				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				rec.obj{t}(i) = obj;
				fprintf('iter %d: obj %f\n',i,obj);
			end
	end
	if transposed
		[rec.rci{t} rec.cci{t}] = deal(cci, rci);
	else
		[rec.rci{t} rec.cci{t}] = deal(rci, cci);
	end
end

% final objective and row/column cluster indicator
[minobj, idx] = min(obj);
rci =  rec.rci{idx};
cci =  rec.cci{idx};
end % main function