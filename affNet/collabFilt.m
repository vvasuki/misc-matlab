function collabFilt()
numFactors = 60;
reg = 10^4;
traceWt = 10^2;
maxiter = 1;
fracEdgesToRemove = .3;
uncertaintyFactor = 5;
bLoad = 0;

dataFile = graph.Constants.ORK_SMALL_NET_10133_75551_gmin2;
load(dataFile, 'socialNet');
[numUsers, numGroups] = size(socialNet.userGroup);

%  [socialNet, A, targetEdgeSet, numPredictions, precisionKatz, completenessKatz, numTargetEdges] = getKatzBaselineAff(fracEdgesToRemove, beta, numIterations);

katz3AffFile = [graph.Constants.DATA_PATH, 'katz3Aff.mat'];
%  save(katz3AffFile , 'A', 'targetEdgeSet', 'numPredictions', 'precisionKatz', 'completenessKatz', 'numTargetEdges', '-v7.3');

load(katz3AffFile);

% Get the Mask matrix for the test missing value set.
numUnknowns =  numPredictions * uncertaintyFactor; 
Mask = getMaskAffCommonNbd(A, numUnknowns);
% or load.
MaskAffFile = [graph.Constants.DATA_PATH, 'MaskAff.mat'];
save(MaskAffFile , 'Mask', 'numUnknowns');
%  load(MaskAffFile);

% Create validation mask.
%  ValidationMaskIndices =  randomizationUtils.Sample.sampleWithoutReplacement(find(Mask == 1), fracMissingValues*numUsers*numGroups);
%  [Training, validationEdgeSet]=getValidationSet(A, fracEdgesToRemove, bLoad);
%  numValidationSetEdges = full(sum(validationEdgeSet));

display ('Compute the optimal User and Group factor matrices');
[UserFactors, S, GroupFactors] = svds(socialNet.userGroup, numFactors);
tic
%  Add a small number to all elements, in case some program has trouble identifying 0's amongst known entries from unknown entries.
socialNet.userGroup = full(socialNet.userGroup) + 1e-6 * ones(numUsers, numGroups);
[UserFactors, GroupFactors] = ALS_knn_U(socialNet.userGroup, Mask, UserFactors*S, GroupFactors, reg, traceWt, double(socialNet.userUser) * double(socialNet.userUser), maxiter);
lrate = 0.001;% What is this???
%  [UserFactors, GroupFactors] = simonfunkMFGL_knn_U(maxiter, reg, traceWt, UserFactors*S, GroupFactors, socialNet.userGroup, Mask, lrate, double(socialNet.userUser) * double(socialNet.userUser));
toc
ScoreMatrix = UserFactors * GroupFactors';
ScoreMatrix = (~Mask) .* ScoreMatrix;

[scoreVector, sortedIndices] = linkPrediction.Predictor.processSimilarityMatrix(A, ScoreMatrix, numPredictions);
[precision, completeness] = statistics.ClassificationEvaluation.checkPrediction(scoreVector, sortedIndices, predictedEdgeIndicesKatz, targetEdgeSet);
fprintf(1, 'precision(CollabFilt): %d\n', precision);
end % function collabFilt

function [Training, validationEdgeSet]=getValidationSet(A, fracEdgesToRemove, bLoad)
    if bLoad == 0
        [Training, validationEdgeSet] = linkPrediction.Predictor.removeRandomEdges(A, fracEdgesToRemove);
    else    
        %  Or load these matrices from a file, instead of calculating them again.
        validationSetFile = [graph.Constants.DATA_PATH, 'trainingValidationSet.mat'];
        load(validationSetFile);
    end
end
