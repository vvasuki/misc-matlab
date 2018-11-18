classdef Predictor
methods(Static=true)
function [precisionRand, sensitivityRand ] = getRandomPrediction(Training, numPredictions, numTargetEdges)
    numTrainingEdges = full(sum(sum(Training))/2);
    [numRows, numCols] = size(Training);
    
    numCorrectEdges = 0;
    if numRows == numCols
        numCorrectEdges = numTargetEdges*numPredictions/(nchoosek(numRows, 2) - numTrainingEdges);
    else
        numCorrectEdges = numTargetEdges*numPredictions/(numRows*numCols - numTrainingEdges);
        
    end
    
    sensitivityRand = numCorrectEdges/numTargetEdges;
    precisionRand = numCorrectEdges/numPredictions;
    
    fprintf(1,'Random predictor: sensitivity, precision: %d, %d \n',sensitivityRand , precisionRand);
end

function comparePerformance(precision, sensitivity, precisionBase, sensitivityBase)
    %  Input: The precision and compelteness scores of the methods being compared.
    display('Comparing performance.')
    %  Compare improvement in precision and sensitivity
    fprintf(1, '  Improvement in precision = %d x \n', precision/precisionBase);
    fprintf(1, '  Improvement in sensitivity = %d x \n', sensitivity/sensitivityBase);

end

function [Training, TargetEdgeSets] = removeRandomEdgesColumnUniform(M, fracEdgesToRemove)
    [numRows, numCols] = size(M);
    TargetEdgeSets = sparse(numRows, numCols);
    Training = sparse(numRows, numCols);
    
    for i=1:numCols
        if ~mod(i,200)
            fprintf('Entity %d\n', i);
        end
        [trainingColumn, targetEdgeSet] = linkPrediction.Predictor.removeRandomEdges(M(:,i), fracEdgesToRemove);
        Training(:,i) = trainingColumn;
        TargetEdgeSets(:,i) = targetEdgeSet;
    end
end

function [Training, TargetEdgeSets] = removeRandomEdgesRowUniform(M, fracEdgesToRemove)
    [Training, TargetEdgeSets] = linkPrediction.Predictor.removeRandomEdgesColumnUniform(M', fracEdgesToRemove);
    Training = Training';
    TargetEdgeSets = TargetEdgeSets';
end

function [Training, targetEdgeSet] = removeRandomEdges(M, fracEdgesToRemove)
    %  Input: M: an adjacency matrix, fracEdgesToRemove. Assumed that M is symmetric.
    %  Output: Training: M with random fracEdgesToRemove edges removed. targetEdgeSet: an indicator vector corresponding to set of edges removed.
    
    [numRowNodes, numColNodes] = size(M);

    % display('Removing random edges')

    if (numRowNodes == numColNodes)
        M = triu(M);
    end
    
    [nzRows, nzCols] = find(M==1);
    nzIndices = find(M==1);
    numEdges = numel(nzRows);
    numEdgesToRetain = floor((1-fracEdgesToRemove)*numEdges);
    
    %  Find sortedIndices of edges to retain.
    edgesToPick = randomizationUtils.Sample.sampleWithoutReplacement(1:numEdges, numEdgesToRetain);
    
    %  Find sortedIndices of edges to remove.
    % display('Finding sortedIndices of edges to remove')
    tmpOnes = ones(numEdges,1);
    tmpOnes(edgesToPick) = 0;
    edgesToRemove = find(tmpOnes);
    clear tmpOnes;
    
    %  Get the matrix Training.
    % display('Making Training')
    %  Local variables.
    tmpRows = nzRows(edgesToPick);
    tmpCols = nzCols(edgesToPick);
    numEdgesPicked = length(edgesToPick);
    Training = sparse(tmpRows, tmpCols, 1, numRowNodes, numColNodes);
    if(numRowNodes == numColNodes)
        Training = Training + Training';
    end
    clear tmpRows, tmpCols;
    
    % display('Making targetEdgeSet')
    edgesToRemoveIndices = nzIndices(edgesToRemove);
    targetEdgeSet = sparse(numRowNodes*numColNodes,1);
    targetEdgeSet(edgesToRemoveIndices) = 1;
end



end
end