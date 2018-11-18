function [precision, completeness] = runLMFAM(numFactors, linkWt, reg, socialNet, A, targetEdgeSet, UserFactors, GroupFactors, L1, L2, numPredictions)

[UserFactors, GroupFactors, L1, L2] = MatFactLBFGSAM(double(socialNet.userUser), A, UserFactors, GroupFactors, L1, L2, linkWt, reg);

scoreMatrix = UserFactors * L1 * GroupFactors';

[scoreVector, sortedIndices] = linkPrediction.Predictor.processSimilarityMatrix(A, scoreMatrix, numPredictions);

[precision, completeness] = statistics.ClassificationEvaluation.checkPrediction(scoreVector, sortedIndices, predictedEdgeIndices, targetEdgeSet);
end
