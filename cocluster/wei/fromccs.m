function [mat] = fromccs(name)

% [mat] = fromccs(name) 
%   convert a sparse matrix in CCS format to Matlab.

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: fromccs.m,v 1.1 2008/05/17 02:10:38 wtang Exp $

fid = fopen(strcat(name, '.bin'), 'rb');
r = fread(fid, 1, 'uint32');
c = fread(fid, 1, 'uint32');
nz = fread(fid, 1, 'uint32');
colptr = double(fread(fid, c+1, 'uint32'));
rowidx = double(fread(fid, nz, 'uint32'));
xx = fread(fid, nz, 'double');
fclose(fid);
mat = getSparse(r, c, rowidx, colptr, xx);