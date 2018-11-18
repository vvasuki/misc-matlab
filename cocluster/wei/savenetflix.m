function savenetflix(mat, name)

% savenetflix(mat, name) 
%   save the netflix data in sparse matrix to CCS binary format.

% Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
% $Id: savenetflix.m,v 1.3 2008/05/18 00:28:32 wtang Exp $

[r,c] = size(mat);
[i,j,xx] = find(mat);
j = cumsum(histc(j, 0:c));

fid = fopen(strcat(name, '.bin'), 'wb');
fwrite(fid, [r c length(xx)], 'uint32');
fwrite(fid, j, 'uint32');
fwrite(fid, i-1, 'uint32');
fwrite(fid, uint8(xx), 'uint8');
fclose(fid);
