% scripts of running experiments based on random split
% the performance of each method is measured by its auc value

clear all;
load 'artificial';
idx = find(triu(A{1},1));    % links
noidx = find(triu(~A{1},1)); % no links
idx = idx(randperm(length(idx)));
noidx = noidx(randperm(length(noidx)));
c = floor(length(idx)/5);
cno = floor(length(noidx)/5);
[n] = size(A{1},1);

% create W where W_ij=1 corresponding to training points
W = cell(3,1);
for i = 1:3
    W{i}.test = zeros(n);
    W{i}.test(idx(1:c*i)) = 1;
    W{i}.test(noidx(1:cno*i)) = 1;
    W{i}.test = logical(W{i}.test + W{i}.test');
    W{i}.train = ~W{i}.test;
    W{i}.train = logical(W{i}.train - diag(diag(W{i}.train)));
    W{i}.train = sparse(W{i}.train);
    W{i}.test = sparse(W{i}.test);
end

% parameter setting
beta = 0.02;
f = 20;
IP = rand(n,f);
IL = rand(f);
IL = (IL+IL')/2;
IIL(1:length(A)) = {IL};
reg = 0.35;
combined = A{1};
for i = 2:length(A)
    combined = combined + A{i};
end

% here we start the experiments
auc = zeros(3,5);
for i = 1:3
    labels = logical(A{1}(W{i}.test));
    nPos = sum(labels == true);
    nNeg = sum(labels == false);
    train = A{1}; train(W{i}.test) = 0;
    comb = combined; comb(W{i}.test) = 0;
    
    % Katz (single)
    disp('katz_single');
    tmp = Katz(train,beta);
    [useless,sidx] = sort(full(tmp(W{i}.test)),'descend');
    auc(i,1) = 1-(sum(find(labels(sidx)))-nPos*(nPos+1)/2)/(nPos*nNeg)
    
    % Katz (combined)
    disp('katz_combined');
    tmp = Katz(comb,beta);
    [useless,sidx] = sort(full(tmp(W{i}.test)),'descend');
    auc(i,2) = 1-(sum(find(labels(sidx)))-nPos*(nPos+1)/2)/(nPos*nNeg)
    
    % SMF (single) unweighted
    disp('smf single');
    [P,L] = MatFactLBFGSAM({spones(train)},{W{i}.train},IP,{IL},reg);
    tmp = P*L{1}*P';
    [useless,sidx] = sort(full(tmp(W{i}.test)),'descend');
    auc(i,3) = 1-(sum(find(labels(sidx)))-nPos*(nPos+1)/2)/(nPos*nNeg)
    
    % SMF (combined) unweighted
    disp('smf combined');
    [P,L] = MatFactLBFGSAM({spones(combined)},{W{i}.train},IP,{IL},reg);
    tmp = P*L{1}*P';
    [useless,sidx] = sort(full(tmp(W{i}.test)),'descend');
    auc(i,4) = 1-(sum(find(labels(sidx)))-nPos*(nPos+1)/2)/(nPos*nNeg)
    
    % LMF (multiple) unweighted
    disp('lmf multiple');
    train = A;
    train{1} = spones(train{1});
    WW(1:length(train))= {W{i}.train};
    [P,L] = MatFactLBFGSAM(train,WW,IP,IIL,reg);
    tmp = P*L{1}*P';
    [useless,sidx] = sort(full(tmp(W{i}.test)),'descend');
    auc(i,5) = 1-(sum(find(labels(sidx)))-nPos*(nPos+1)/2)/(nPos*nNeg)
end
% save(strcat(directory,'auc_',suffix),'auc');
