
%INPUT: RESPONSE Y, PREDICTOR X, PENALTY LAMBDA
%OUTPUT: NEIGHBORHOOD NBRLASSO, COEFFICIENTS BETA 
function [nbrlasso,beta] = logisticRegBoyd(y,X,lambda,lcnst,tolrcnst)

n = size(X,1);
p = size(X,2);
%TOLR = 0.02;
TOLR = tolrcnst * lambda;

Xfile = sprintf('/tmp/X-%d-%d-%.2f',p,n,lcnst);
yfile = sprintf('/tmp/y-%d-%d-%.2f',p,n,lcnst);
modelfile = sprintf('/tmp/model-%d-%d-%.2f',p,n,lcnst);

mmwrite(Xfile,X);
mmwrite(yfile,y);

system(sprintf('/public/linux/graft/optimization/l1_logreg-0.8.2-i686-pc-linux-gnu/l1_logreg_train -q -s %s %s %f %s',Xfile,yfile,lambda,modelfile));
model_lg = full(mmread(modelfile));
beta = model_lg(2:length(model_lg));

nbrlasso = abs(beta) >= TOLR;
numnbr = sum(nbrlasso);
%minnbr = min(beta(abs(beta) >= TOLR))
