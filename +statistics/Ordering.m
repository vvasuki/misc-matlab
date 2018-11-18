classdef Ordering
methods(Static=true)
function [precision, sensitivity, specificity] = getPerformance(scoreGenerator, ItemsToIgnore, targetItemSet, parameters, numPredictions)
    ScoreMatrix = scoreGenerator(parameters);
    [scoreVector, sortedIndices] = statistics.Ordering.processSimilarityMatrix(ItemsToIgnore, ScoreMatrix);
    
    [precision, sensitivity, specificity] = statistics.ClassificationEvaluation.checkPrediction(scoreVector, sortedIndices, numPredictions, targetItemSet);
    fprintf('parameters: ');
    fprintf('%d ',parameters');
    fprintf(' precision: %d, sensitivity: %d, specificity: %d\n', precision, sensitivity, specificity);
end

function [precision, sensitivity, specificity] = getAveragePerformance(scoreGenerator, ItemsToIgnore, TargetItemSets, parameters, numPredictions)
    import statistics.*;
    [precision, sensitivity, specificity] = ClassificationEvaluation.getAveragePerformance(scoreGenerator(parameters), ItemsToIgnore, TargetItemSets, numPredictions);
    fprintf('parameters: ');
    fprintf('%d ',parameters');
    fprintf(' precision: %d, sensitivity: %d, specificity: %d\n', precision, sensitivity, specificity);
end

function [precision, sensitivity, specificity] = getAvgPerformance(scoreGenerator, ItemsToIgnore, targetItemSet, parameters, numPredictions)
    ScoreMatrix = scoreGenerator(parameters);
    [scoreVector, sortedIndices] = statistics.Ordering.processSimilarityMatrix(ItemsToIgnore, ScoreMatrix);
    
    [precision, sensitivity, specificity] = statistics.ClassificationEvaluation.checkPredictions(scoreVector, sortedIndices, numPredictions, targetItemSet);
    fprintf('parameters: ');
    fprintf('%d ',parameters');
    fprintf(' precision: %d, sensitivity: %d, specificity: %d\n', precision, sensitivity, specificity);
end

function [sortedScores, sortedIndices] = processSimilarityMatrix(ItemsToIgnore, ScoreMatrix)
%  Input: 
%  ItemsToIgnore: 1 is for items to ignore.
%  numPredictions: number of edges to output.
%  Logic for using the mask matrix is implemented only for rectangular matrices.

    [numRowNodes, numColNodes] = size(ScoreMatrix);
    if nargin < 3
        ItemsToIgnore = sparse(numRowNodes, numColNodes);
    end
            
    if (numRowNodes == numColNodes) 
        ScoreMatrix = triu(ScoreMatrix);
    end
    
    scoreVector = full(ScoreMatrix(:));
    
    scoreVector = statistics.Ordering.setLowScoresIgnoreItems(scoreVector, ItemsToIgnore);
    
%      display('Sorting')
%      tic
    [sortedScores sortedIndices] = sort(full(scoreVector),'descend');
%      toc
end

function scoreVector = setLowScoresIgnoreItems(scoreVector, ItemsToIgnore)
%   Set scores for items which should never be considered for prediction to -Inf.
    [numRowNodes, numColNodes] = size(ItemsToIgnore);
    if(nargin>2)
%      ItemsToIgnore has been passed
        tmpNzIndices = find(ItemsToIgnore(:));
%          tic
        scoreVector(tmpNzIndices) = -Inf;
%          toc
    end
    
    if (numRowNodes == numColNodes) 
%          display('Setting scores corresponding to self-loops to 0')
        I = speye(numRowNodes);
        tmpNzIndices = find(I(:));
%          display('.. Found necessary sortedIndices')
%          tic
        scoreVector (tmpNzIndices) = -Inf;
%          toc
    end

end


function topPredictions = getTopPredictions(numPredictions, sortedScoresAll, sortedIndicesAll)
%          ASSUMPTION: This assumes that items which should NEVER be predicted have score = -INF
    import statistics.*;
    [sortedScoresTrimmed, sortedIndices] = ClassificationEvaluation.trimSortedScoresItems(numPredictions + 1, sortedScoresAll, sortedIndicesAll);
    
    cutoff = sortedScoresTrimmed(end);
    firstElementInEq = find(sortedScoresTrimmed == cutoff, 1, 'first');
    gt = sortedIndices(1: firstElementInEq - 1);
    topPredictions = gt;
end

function topPredictionSets=getTopPredictionSets(numPredictions, ScoreMatrix, ItemsToIgnore)
    % Iterates over rows.
    import statistics.*;
    topPredictionSets = {};
    
    [numSets, numItemsPerSet] = size(ScoreMatrix);
    for i=1:numSets
        [sortedScores, sortedIndices] =     Ordering.processSimilarityMatrix(ItemsToIgnore(i,:), ScoreMatrix(i,:));
        topPredictionSets{i} = Ordering.getTopPredictions(numPredictions, sortedScores, sortedIndices);
    end
end

function [precision, sensitivity, specificity] = predictionFromSimilarity(ItemsToIgnore, ScoreMatrix, numPredictions, targetEdgeSet)
    [predictedEdgeIndices, scoreVector, sortedIndices] = statistics.Ordering.processSimilarityMatrix(ItemsToIgnore, ScoreMatrix, numPredictions);
    [precision, sensitivity, specificity] = statistics.ClassificationEvaluation.checkPrediction(ItemsToIgnore, predictedEdgeIndices, targetEdgeSet);
end

function testProcessSimilarityMatrix()
    ItemsToIgnore = zeros(10, 1);
    ScoreMatrix = (1:10);
    ScoreMatrix(4:10) = 0;
    numPredictions = 5
    [scoreVector, sortedIndices] = statistics.Ordering.processSimilarityMatrix(ItemsToIgnore, ScoreMatrix, numPredictions)
end


end
end