
%INPUT: 
%	numNodes: NUMBER OF VARIABLES 
%	numSamples: NUMBER OF SAMPLES 
%	lcnst: penalty lambda = lcnst x asymptotic-lambda 
%	tolrcnst: threshold TOLR = tolrcnst x lambda
%OUTPUT: 
%	probsucc: PROB. OF SUCCESS
%	fracdisagree:  FRACTION OF DISAGREEING EDGES
%  Eg: structSearchInstance(10,100,0.1,0.1)

function [probsucc,fracdisagree] = structSearchInstance(numNodes,numSamples,lcnst,tolrcnst)

import graph.*;
import probabilisticModels.*;

topology = 'star'
%degmax = ceil(log(numNodes));
%degmax = numNodes - 1;
%degmax = 5;
%dcoup = 5;

degmax = ceil(0.1 * numNodes)
dcoup = min(0.1,2.5/degmax)
edgeWeightSign = 'mixed'

mtype = java.util.Hashtable;
mtype.put('numNodes',numNodes);
mtype.put('degmax',degmax);
mtype.put('topology',topology);
mtype.put('edgeWeightSign',edgeWeightSign);
mtype.put('coupStrength',dcoup);

nexp = 1
lambdaAsymp = lcnst * sqrt(log(numNodes)/numSamples)

lambda = lambdaAsymp

%[graph,X] = getSearchInstance(mtype,numSamples,lambda);

graph = getGraphModel(mtype);
truegraph = (graph ~= 0);

%  	Begin new test code
[adjMatrix, nodePotentials, edgePotentials] = DiscreteRVGraphicalModel.getBinaryRVGraphicalModel(topology, numNodes, degmax, dcoup, edgeWeightSign)
truegraph = adjMatrix;
%  	End new test code


numTrials = 100;
numMethods = 5;

probsucc = zeros(numMethods,1);
fracdisagree = zeros(numMethods,1);
graphret = cell(numMethods,1);
for(m = 1:numMethods)
	graphret{m} = zeros(numNodes);
end
tmp = zeros(numMethods,1);
tmpos = zeros(numMethods,1);
tmpneg = zeros(numMethods,1);
betamat = cell(numTrials,1);

for(numT = 1:numTrials)
	numT
	
	timer = Timer();
	X = getSamples(graph,numSamples,topology);
	timer = timer.endTimer();
	fprintf('Ones: %d -Ones: %d\n', sum(sum(X == 1)), sum(sum(X == -1)));
	
	
%  	Begin new test code
	timer = Timer();
        bIsTree = Graph.isTree(topology);
        Samples = DiscreteRVGraphicalModel.getSamples(adjMatrix, nodePotentials, edgePotentials, bIsTree, numSamples);
	Samples(Samples == 2) = -1;
	X = Samples';
	timer = timer.endTimer();
	fprintf('Ones: %d -Ones: %d\n', sum(sum(X == 1)), sum(sum(X == -1)));
%  	End new test code
	
	disp('L1')
	graphL1 = zeros(numNodes);
	for s = 1:numNodes
%  		s
		Vbs = setdiff(1:numNodes,[s]);
		[thetanbr1,betamat{numT}] = logisticRegBoyd(X(:,s),X(:,Vbs),lambda,lcnst,tolrcnst);
		graphL1(Vbs,s) = thetanbr1;
	end	
	%Combining neighborhood estimates via AND, OR
	graphret{1} = ((graphL1 + graphL1') == 2);
	graphret{2} = ((graphL1 + graphL1') > 0);
	
	disp('PC')
	%graphret{3} = pcMatWrapper(X);
	disp('CL')
	[graphret{4},graphret{5}] = chowliu(X,degmax);
		
	
	for(m = 1:numMethods)
		tmpos(m) = sum(sum(graphret{m} > truegraph));
		tmpneg(m) = sum(sum(graphret{m} < truegraph));
		tmp(m) = sum(sum(graphret{m} ~= truegraph));
		probsucc(m) = probsucc(m) + (tmp(m) == 0);
		fracdisagree(m) = fracdisagree(m) + tmp(m);
	end
end

tmpos
tmpneg
probsucc = probsucc/numTrials
fracdisagree = fracdisagree/numTrials
filestr = sprintf('outputx/ss-%d-%d-%.2f.mat',numNodes,numSamples,lcnst);
save(filestr,'probsucc','fracdisagree','betamat');

