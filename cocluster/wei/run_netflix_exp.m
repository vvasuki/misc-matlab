function run_netflix_exp(algorithm, basis)
% Scripts for runing the experiments of Bregman co-clustering on Netflix
% data set.
% 

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: run_netflix_exp.m,v 1.9 2008/05/18 00:28:32 wtang Exp $

error(nargchk(2, 2, nargin));
train = loadnetflix('train');
probe = loadnetflix('probe');
train = train - probe;
nrc = [2 4 8 16 32];
ncc = [2 4 8 16 32];
ntrials = length(nrc);

rmse_rec = zeros(1,ntrials);
exp_rec = cell(1,ntrials);

switch (algorithm)
	case 'ecc'
		for i = 1:ntrials
			fprintf('%s(basis %d): #rc/#cc = %d/%d\n',algorithm, basis, nrc(i), ncc(i));
			tic
			[rci,cci,mobj,exp_rec{i}] = ecc(train,nrc(i),ncc(i),'basis',basis,'maxi',100);
			rmse_rec(i) = rmse(train,probe,rci,cci,basis);
			toc
			fprintf('rmse=%f\n',rmse_rec(i));
			save(strcat(algorithm,dec2base(basis,10),'result'),'rmse_rec','exp_rec');
		end
	case 'icc'
		for i = 1:ntrials
			fprintf('%s(basis %d): #rc/#cc = %d/%d\n',algorithm, basis, nrc(i), ncc(i));
			tic
			[rci,cci,mobj,exp_rec{i}] = icc(train,nrc(i),ncc(i),'basis',basis,'maxi',100);
			rmse_rec(i) = rmse(train,probe,rci,cci,basis);
			toc
			fprintf('rmse=%f\n',rmse_rec(i));
			save(strcat(algorithm,dec2base(basis,10),'result'),'rmse_rec','exp_rec');
		end
	otherwise
		error('Unrecognized algorithm type');
end
