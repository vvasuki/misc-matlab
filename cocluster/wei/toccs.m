function toccs(mat, name)

% toccs(mat, name) 
%   convert a sparse matrix in Matlab to CCS binary format.

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: toccs.m,v 1.3 2008/05/17 02:10:38 wtang Exp $

[r,c] = size(mat);
[i,j,xx] = find(mat);
j = cumsum(histc(j, 0:c));

fid = fopen(strcat(name, '.bin'), 'wb');
fwrite(fid, [r c length(xx)], 'uint32');
fwrite(fid, j, 'uint32');
fwrite(fid, i-1, 'uint32');
fwrite(fid, xx, 'double');
fclose(fid);
