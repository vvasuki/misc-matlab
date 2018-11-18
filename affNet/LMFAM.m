numFactors = 30;
linkWt = 10^-2;
reg = 10^3;
fracEdgesToRemove = .3;
beta = 0.5;
numIterations = 3;

%  Get baseline
katz3AffFile = [graph.Constants.DATA_PATH, 'katz3Aff.mat'];
%  %  save(katz3AffFile , 'A', 'targetEdgeSet', 'numPredictions', 'precisionKatz', 'completenessKatz', 'numTargetEdges', '-v7.3');
%  
dataFile = graph.Constants.ORK_SMALL_NET_10133_75551_gmin2;
load(dataFile, 'socialNet');
load(katz3AffFile);

% Create validation and training matrices.
%  [Training, validationEdgeSet] = linkPrediction.Predictor.removeRandomEdges(A, fracEdgesToRemove);
validationSetFile = [graph.Constants.DATA_PATH, 'trainingValidationSet.mat'];
%  save(validationSetFile , 'Training', 'validationEdgeSet', '-v7.3');

% Or load these matrices from a file, instead of calculating them again.
load(validationSetFile);


[numUsers, numGroups] = size(socialNet.userGroup);
numValidationSetEdges = full(sum(validationEdgeSet));


file = fopen([graph.Constants.DATA_PATH, 'LMFAMruns'], 'w');
%  UserFactors = rand(numUsers, numFactors);
%  GroupFactors = rand(numGroups, numFactors);
[UserFactors,S,GroupFactors] = svds(Training, numFactors);
%  L1 = rand(numFactors);
L1 = S(1:numFactors, 1:numFactors);
L2 = rand(numFactors);
L2 = (L2 + L2')/10;

for numFactors=40:30:100
%  for linkWt=0.1:0.4:1.0
%  for regPow=1:4:9
linkWt=0.4;
regPow=4;
    [precision, completeness] = runLMFAM(numFactors, linkWt, 10^regPow, socialNet, Training, validationEdgeSet, UserFactors, GroupFactors, L1, L2, numValidationSetEdges );
    fprintf(file, 'numFactors: %d, linkWt: %f, regPow: %d, precision: %f\n', numFactors, linkWt, regPow, precision);
%  end
%  end
end

%  Use the best linkWt, regPow and numFactors from validation.
numFactors = 60;
linkWt = 0.5;
regPow = 4;
%  Check performance against Test set.
% Compare performance.
[precision, completeness] = runLMFAM(numFactors, linkWt, 10^regPow, socialNet, A, testEdgeSet, UserFactors, GroupFactors, L1, L2, numPredictions);
comparePerformance(precision, completeness, precisionKatz, completenessKatz);
