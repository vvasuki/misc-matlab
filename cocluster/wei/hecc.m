function [rch, cch, objs] = hecc(mat, nlevels, varargin)

% HECC Hierarchical Bregman co-clustering algorithm using squared Euclidean
% distance.
%
% [rch, cch, objs] = hecc(mat, nlevels, [[argID,] value, ...])
%
% Examples:
%	[rch, cch] = hecc(mat, nlevels)
%	[rch, cch, objs] = hecc(mat, nlevels)
%
% INPUT:
%	mat     (matrix) 2-D matrix with any real values, size nr by nc
%	nlevels (scalar) number of levels in the hierarchy (default = 5)
%
%	Valid argument IDs and corresponding values are:
%	  'maxi'      (scalar) maximum number of iterations (default = 50)
%	  'relerr'    (scalar) relative stopping criterion (default = 1e-5)
%	  'basis'     (scalar) basis of bregman co-clustering (default = 5)
%	  'weight'    (matrix) the weight matrix, same size of mat
%	  'nrc'       (vector) the num of row clusters for each par row cluster
%	                       size 1 by nlevels
%	  'ncc'       (vector) the num of col clusters for each par col cluster
%	                       size 1 by nlevels
%
% OUTPUT:
%	rch    (matrix) resulting row cluster hierarchy, size nr by nlevels
%	cch    (matrix) resulting col cluster hierarchy, size nc by nlevels
%	objs   (vector) the objective in each level of the hierarchy
%
% See also: ecc

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: hecc.m,v 1.16 2008/05/18 00:28:32 wtang Exp $


% check input arguments
error(nargchk(2, 14, nargin));

% size of the input matrix
[nr,nc] = size(mat);

% normalize the input matrix
[I,J,xx] = find(mat);
mat = sparse(I,J,xx-mean(xx),nr,nc);
clear I J xx;

% default input arguments
maxi = 50;
relerr = 1e-5;
basis = 5;
wgt = spones(mat);

% by default, each cluster in the parent level is splitted into two
% clusters in the current level
nrc = 2*ones(1,nlevels);
ncc = 2*ones(1,nlevels);

% parse varargin
i = 1;
while i <= length(varargin)
	if ischar(varargin{i})
		switch varargin{i}
			case 'maxi'
				i = i+1; maxi = varargin{i};
			case 'relerr'
				i = i+1; relerr = varargin{i};
			case 'basis'
				i = i+1; basis = varargin{i};
			case 'weight'
				i = i+1; wgt = varargin{i};
			case 'nrc'
				i = i+1; nrc = varargin{i};
			case 'ncc'
				i = i+1; ncc = varargin{i};
			otherwise
				error('Ignoring invalid argID %s', varargin{i});
		end
	else
		error('Ignoring invalid argument #%d', i+2);
	end
	i = i+1;
end

% check the input arguments
if size(wgt) ~= size(mat)
	error('The weight matrix should be in the same size of data matrix');
end
if basis < 1 || basis > 6
	error('The basis should be in the range of 1 to 6');
end
if length(nrc) ~= nlevels
	error('The length of nrc must be %d!', nlevels);
end
if length(ncc) ~= nlevels
	error('The length of ncc must be %d!', nlevels);
end

% pre-compute some constants
if basis == 6
	orig_mat = mat;
end
mat = wgt.*mat;
Zm = full(sum(sum(mat))/sum(sum(wgt)));     % global mean
Zr = full(spdotdiv(sum(mat,2),sum(wgt,2))); % row mean
Zc = full(spdotdiv(sum(mat,1),sum(wgt,1))); % column mean

objs = zeros(1, nlevels);    % the objective in each level
rch = zeros(nr, nlevels);
cch = zeros(nc, nlevels);

% hierarchical co-clustering in nlevels
for t = 1:nlevels
	% make sure the ncc is always no less than nrc
	transposed = nrc(t) > ncc(t);
	if transposed
		mat = mat';
		wgt = wgt';
		if basis == 6
			orig_mat = orig_mat';
		end
		[nrc(t),ncc(t),Zr,Zc,rch,cch,nr,nc] = deal(ncc(t),nrc(t),Zc',Zr',cch,rch,nc,nr);
	end
	fprintf('level %d:\n',t);
	% the row/col cluster indicator in previous level
	if t == 1
		prci = ones(nr,1);
		pcci = ones(nc,1);
	else
		prci = rch(:,t-1);
		pcci = cch(:,t-1);
	end
	nprci = length(unique(prci)); % num of row clusters in previous level
	npcci = length(unique(pcci)); % num of col clusters in previous level
	% initialize the row/col clusters based on the previous one
	rci = zeros(nr,1);
	cci = zeros(nc,1);
	for r = 1:nprci
		rc = logical(prci==r);
		tmp = mod((1:sum(rc))',nrc(t))+1;
		rci(rc) = tmp(randperm(length(tmp)))+(r-1)*nrc(t);
	end
	for c = 1:npcci
		cc = logical(pcci==c);
		tmp = mod((1:sum(cc))',ncc(t))+1;
		cci(cc) = tmp(randperm(length(tmp)))+(c-1)*ncc(t);
	end
	R = spconvert([(1:nr)' rci ones(nr,1)]);
	C = spconvert([(1:nc)' cci ones(nc,1)]);
	switch basis
		case 1
			for i = 1:maxi
				Zh = full(spdotdiv(sum(mat*C,1),sum(wgt*C,1)));
				ZrowRed = spdotdiv(mat*C,wgt*C) + Zm ...
					- spdotdiv(wgt*diag(sparse(Zh*C'))*C,wgt*C);
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZrowVRed = Zg(:,ones(1,length(unique(cci))));
				% basis 1 row update
				for r = 1:nprci
					rc = logical(prci==r);
					idx = 1+nrc(t)*(r-1):nrc(t)*r;
					% compute row assignement distance matrix
					D = sqeuclidean(ZrowRed(rc,:),ZrowVRed(idx,:),full(wgt(rc,:)*C));
					% update the row cluster indicator
					[minvals, tmp] = min(D, [], 2);
					rci(rc) = tmp + (r-1)*nrc(t);
					R(rc,idx) = spconvert([(1:sum(rc))' tmp ones(sum(rc),1)]);
				end
				% basis 1 column update
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZcolRed = spdotdiv(R'*mat,R'*wgt) + Zm ...
					- spdotdiv(R'*diag(sparse(R*Zg))*wgt,R'*wgt);
				Zh = full(spdotdiv(sum(mat*C,1),sum(wgt*C,1)));
				ZcolVRed = Zh(ones(1,length(unique(rci))),:);
				for c = 1:npcci
					cc = logical(pcci==c);
					idx = 1+ncc(t)*(c-1):ncc(t)*c;
					% compute column assignment distance matrix
					D = sqeuclidean(ZcolRed(:,cc)',ZcolVRed(:,idx)',full(wgt(:,cc)'*R));
					% update the column cluster indicator
					[minvals, tmp] = min(D, [], 2);
					cci(cc) = tmp + (c-1)*ncc(t);
					C(cc,idx) = spconvert([(1:sum(cc))' tmp ones(sum(cc),1)]);
				end
				% current objective value
				obj = eucobj(mat, wgt, rci, cci, Zm, Zr, Zc, 1);
				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				fprintf('iter %d: obj %f\n',i,obj);
			end
		case 2
			for i = 1:maxi
				% basis 2 row update
				ZrowRed = full(spdotdiv(mat*C,wgt*C));
				ZrowVRed = full(spdotdiv(R'*mat*C,R'*wgt*C));
				for r = 1:nprci
					rc = logical(prci==r);
					idx = 1+nrc(t)*(r-1):nrc(t)*r;
					% compute row assignement distance matrix
					D = sqeuclidean(ZrowRed(rc,:),ZrowVRed(idx,:),full(wgt(rc,:)*C));
					% update the row cluster indicator
					[minvals, tmp] = min(D, [], 2);
					rci(rc) = tmp + (r-1)*nrc(t);
					R(rc,idx) = spconvert([(1:sum(rc))' tmp ones(sum(rc),1)]);
				end
				% basis 2 column update
				ZcolRed = full(spdotdiv(R'*mat,R'*wgt));
				ZcolVRed = full(spdotdiv(R'*mat*C,R'*wgt*C));
				for c = 1:npcci
					cc = logical(pcci==c);
					idx = 1+ncc(t)*(c-1):ncc(t)*c;
					% compute column assignment distance matrix
					D = sqeuclidean(ZcolRed(:,cc)',ZcolVRed(:,idx)',full(wgt(:,cc)'*R));
					% update the column cluster indicator
					[minvals, tmp] = min(D, [], 2);
					cci(cc) = tmp + (c-1)*ncc(t);
					C(cc,idx) = spconvert([(1:sum(cc))' tmp ones(sum(cc),1)]);
				end
				% current objective value
				obj = eucobj(mat, wgt, rci, cci, Zm, Zr, Zc, 2);
				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				fprintf('iter %d: obj %f\n',i,obj);
			end
		case 3
			for i = 1:maxi
				% basis 3 row update
				ZrowRed = full(spdotdiv(mat*C,wgt*C) ...
					- spdotdiv(diag(sparse(Zr))*wgt*C,wgt*C));
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZrowVRed = spdotdiv(R'*mat*C,R'*wgt*C) ...
					- Zg(:,ones(1,length(unique(cci))));
				for r = 1:nprci
					rc = logical(prci==r);
					idx = 1+nrc(t)*(r-1):nrc(t)*r;
					% compute row assignement distance matrix
					D = sqeuclidean(ZrowRed(rc,:),ZrowVRed(idx,:),full(wgt(rc,:)*C));
					% update the row cluster indicator
					[minvals, tmp] = min(D, [], 2);
					rci(rc) = tmp + (r-1)*nrc(t);
					R(rc,idx) = spconvert([(1:sum(rc))' tmp ones(sum(rc),1)]);
				end
				% basis 3 column update
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZcolRed = full(spdotdiv(R'*mat,R'*wgt) ...
					- spdotdiv(R'*diag(sparse(Zr))*wgt,R'*wgt) ...
					+ spdotdiv(R'*diag(sparse(R*Zg))*wgt,R'*wgt));
				ZcolVRed = full(spdotdiv(R'*mat*C,R'*wgt*C));
				for c = 1:npcci
					cc = logical(pcci==c);
					idx = 1+ncc(t)*(c-1):ncc(t)*c;
					% compute column assignment distance matrix
					D = sqeuclidean(ZcolRed(:,cc)',ZcolVRed(:,idx)',full(wgt(:,cc)'*R));
					% update the column cluster indicator
					[minvals, tmp] = min(D, [], 2);
					cci(cc) = tmp + (c-1)*ncc(t);
					C(cc,idx) = spconvert([(1:sum(cc))' tmp ones(sum(cc),1)]);
				end
				% current objective value
				obj = eucobj(mat, wgt, rci, cci, Zm, Zr, Zc, 3);
				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				fprintf('iter %d: obj %f\n',i,obj);
			end
		case 4
			for i = 1:maxi
				% basis 4 row update
				Zh = full(spdotdiv(sum(mat*C,1),sum(wgt*C,1)));
				ZrowRed = full(spdotdiv(mat*C,wgt*C) ...
					- spdotdiv(wgt*diag(sparse(Zc))*C,wgt*C) ...
					+ spdotdiv(wgt*diag(sparse(Zh*C'))*C,wgt*C));
				ZrowVRed = full(spdotdiv(R'*mat*C,R'*wgt*C));
				for r = 1:nprci
					rc = logical(prci==r);
					idx = 1+nrc(t)*(r-1):nrc(t)*r;
					% compute row assignement distance matrix
					D = sqeuclidean(ZrowRed(rc,:),ZrowVRed(idx,:),full(wgt(rc,:)*C));
					% update the row cluster indicator
					[minvals, tmp] = min(D, [], 2);
					rci(rc) = tmp + (r-1)*nrc(t);
					R(rc,idx) = spconvert([(1:sum(rc))' tmp ones(sum(rc),1)]);
				end
				% basis 4 column update
				ZcolRed = full(spdotdiv(R'*mat,R'*wgt) ...
					- spdotdiv(R'*wgt*diag(sparse(Zc)),R'*wgt));
				Zh = full(spdotdiv(sum(mat*C,1),sum(wgt*C,1)));
				ZcolVRed = spdotdiv(R'*mat*C,R'*wgt*C) ...
					- Zh(ones(1,length(unique(rci))),:);
				for c = 1:npcci
					cc = logical(pcci==c);
					idx = 1+ncc(t)*(c-1):ncc(t)*c;
					% compute column assignment distance matrix
					D = sqeuclidean(ZcolRed(:,cc)',ZcolVRed(:,idx)',full(wgt(:,cc)'*R));
					% update the column cluster indicator
					[minvals, tmp] = min(D, [], 2);
					cci(cc) = tmp + (c-1)*ncc(t);
					C(cc,idx) = spconvert([(1:sum(cc))' tmp ones(sum(cc),1)]);
				end
				% current objective value
				obj = eucobj(mat, wgt, rci, cci, Zm, Zr, Zc, 4);
				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				fprintf('iter %d: obj %f\n',i,obj);
			end
		case 5
			for i = 1:maxi
				% basis 5 row update
				Zh = full(spdotdiv(sum(mat*C,1),sum(wgt*C,1)));
				ZrowRed = full(spdotdiv(mat*C,wgt*C) ...
					- spdotdiv(diag(sparse(Zr))*wgt*C,wgt*C) ...
					- spdotdiv(wgt*diag(sparse(Zc))*C,wgt*C) ...
					+ spdotdiv(wgt*diag(sparse(Zh*C'))*C,wgt*C));
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZrowVRed = spdotdiv(R'*mat*C,R'*wgt*C) ...
					- Zg(:,ones(1,length(unique(cci))));
				for r = 1:nprci
					rc = logical(prci==r);
					idx = 1+nrc(t)*(r-1):nrc(t)*r;
					% compute row assignement distance matrix
					D = sqeuclidean(ZrowRed(rc,:),ZrowVRed(idx,:),full(wgt(rc,:)*C));
					% update the row cluster indicator
					[minvals, tmp] = min(D, [], 2);
					rci(rc) = tmp + (r-1)*nrc(t);
					R(rc,idx) = spconvert([(1:sum(rc))' tmp ones(sum(rc),1)]);
				end
				% basis 5 column update
				Zg = full(spdotdiv(sum(R'*mat,2),sum(R'*wgt,2)));
				ZcolRed = full(spdotdiv(R'*mat,R'*wgt) ...
					- spdotdiv(R'*diag(sparse(Zr))*wgt,R'*wgt) ...
					- spdotdiv(R'*wgt*diag(sparse(Zc)),R'*wgt) ...
					+ spdotdiv(R'*diag(sparse(R*Zg))*wgt,R'*wgt));
				Zh = full(spdotdiv(sum(mat*C,1),sum(wgt*C,1)));
				ZcolVRed = spdotdiv(R'*mat*C,R'*wgt*C) ...
					- Zh(ones(1,length(unique(rci))),:);
				for c = 1:npcci
					cc = logical(pcci==c);
					idx = 1+ncc(t)*(c-1):ncc(t)*c;
					% compute column assignment distance matrix
					D = sqeuclidean(ZcolRed(:,cc)',ZcolVRed(:,idx)',full(wgt(:,cc)'*R));
					% update the column cluster indicator
					[minvals, tmp] = min(D, [], 2);
					cci(cc) = tmp + (c-1)*ncc(t);
					C(cc,idx) = spconvert([(1:sum(cc))' tmp ones(sum(cc),1)]);
				end
				% current objective value
				obj = eucobj(mat, wgt, rci, cci, Zm, Zr, Zc, 5);
				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				fprintf('iter %d: obj %f\n',i,obj);
			end
		case 6
			for i = 1:maxi
				% basis 6 row update
				Zrh = spdotdiv(mat*C,wgt*C);
				ZrowVar = full(spdotdiv(R'*mat,R'*wgt) ...
					- spdotdiv(R'*mat*C,R'*wgt*C)*C');
				for r = 1:nprci
					rc = logical(prci==r);
					idx = 1+nrc(t)*(r-1):nrc(t)*r;
					% compute row assignement distance matrix
					D = sqeuclidean6(orig_mat(rc,:), Zrh(rc,:), cci, ...
						ZrowVar(idx,:), wgt(rc,:));
					% update the row cluster indicator
					[minvals, tmp] = min(D, [], 2);
					rci(rc) = tmp + (r-1)*nrc(t);
					R(rc,idx) = spconvert([(1:sum(rc))' tmp ones(sum(rc),1)]);
				end
				% basis 6 column update
				Zgc = spdotdiv(R'*mat,R'*wgt);
				ZcolVar = full(spdotdiv(mat*C,wgt*C) ...
					- R*spdotdiv(R'*mat*C,R'*wgt*C));
				for c = 1:npcci
					cc = logical(pcci==c);
					idx = 1+ncc(t)*(c-1):ncc(t)*c;
					% compute column assignment distance matrix
					D = sqeuclidean6(orig_mat(:,cc)', Zgc(:,cc)', rci, ...
						ZcolVar(:,idx)', wgt(:,cc)');
					% update the column cluster indicator
					[minvals, tmp] = min(D, [], 2);
					cci(cc) = tmp + (c-1)*ncc(t);
					C(cc,idx) = spconvert([(1:sum(cc))' tmp ones(sum(cc),1)]);
				end
				% current objective value
				obj = eucobj(mat, wgt, rci, cci, Zm, Zr, Zc, 6);
				if i > 1
					if old_obj-obj <= old_obj*relerr
						break;
					end
				end
				old_obj = obj;
				fprintf('iter %d: obj %f\n',i,obj);
			end
	end
	[rch(:,t) cch(:,t)] = deal(rci, cci);
	objs(t) = obj;
	if transposed
		mat = mat';
		wgt = wgt';
		if basis == 6
			orig_mat = orig_mat';
		end
		[nrc(t),ncc(t),Zr,Zc,rch,cch,nr,nc] = deal(ncc(t),nrc(t),Zc',Zr',cch,rch,nc,nr);
	end
	save(strcat('hecc',dec2base(basis,10),'result'),'rch','cch','objs');
end
end % main function