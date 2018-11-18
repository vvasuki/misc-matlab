function [mat] = loadnetflix(name)

% [mat] = loadnetflix(name) 
%   (NOTE: For Netflix data only)
%   load the netflix data from sparse matrix in CCS format to Matlab.

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: loadnetflix.m,v 1.3 2008/05/18 00:28:32 wtang Exp $

fid = fopen(strcat(name, '.bin'), 'rb');
r = fread(fid, 1, 'uint32');
c = fread(fid, 1, 'uint32');
nz = fread(fid, 1, 'uint32');
colptr = fread(fid, c+1, 'uint32=>double');
rowidx = fread(fid, nz, 'uint32=>double');
xx = fread(fid, nz, 'uint8=>double');
fclose(fid);
mat = getsparse(r, c, rowidx, colptr, xx);