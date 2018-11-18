function dat = laplacian_normalize(dat)

d = sum(dat);
d(d==0) = 1e-6;
% idx = find(d);
% d(idx) = d(idx).^-.5;
d = d.^-.5;
dat = diag(d)*dat*diag(d);

end % main function