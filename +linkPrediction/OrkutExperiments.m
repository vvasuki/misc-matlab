classdef OrkutExperiments
methods(Static = true)
function [socialNet, A, targetEdgeSet, numPredictions, katzScoreMatrix, precisionKatz, completenessKatz, numTargetEdges] = getKatzBaseline(fracEdgesToRemove, beta, numIterations, socialNet)
%      Input: If socialNet is not passed, the method will load it from some data file.
%      Output: Load a social network, measure the performance of the Truncated Katz technique as a baseline, and return the scores.
%  
    if(nargin < 4)
        dataFile = graph.Constants.ORK_SMALL_NET_10133_75551_gmin2;
        load(dataFile, 'socialNet');
    end
    %  clear socialNet;
    
    numUsers = size(socialNet.userUser, 1);
    
    [A, targetEdgeSet] = linkPrediction.Predictor.removeRandomEdges(socialNet.userUser, fracEdgesToRemove);
    
    numTrainingEdges = full(sum(sum(A))/2);
    numTargetEdges = full(sum(targetEdgeSet));
    numPredictions =  numTargetEdges;
    
    fprintf(1, 'training edges: %d, target edges: %d, predictions: %d \n', numTrainingEdges, numTargetEdges, numPredictions);
    
    display('Calculating Truncated Katz')
    katzScoreMatrix = linkPrediction.Katz.KatzApprox(A, numIterations, beta);
    display('Got Katz matrix')
    [scoreVector, sortedIndices] = linkPrediction.Predictor.processSimilarityMatrix(A, katzScoreMatrix, numPredictions);
    
    [precisionKatz, completenessKatz] = statistics.ClassificationEvaluation.checkPrediction(scoreVector, sortedIndices, predictedEdgeIndicesKatz, targetEdgeSet);
    
    %  How good is the Katz prediction?
    fprintf(1, 'Katz: precision, completeness = %d %d \n', precisionKatz, completenessKatz);
end

end
end