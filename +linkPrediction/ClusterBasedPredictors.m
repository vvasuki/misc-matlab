classdef ClusterBasedPredictors
methods(Static=true)
function [precisionClRand, completenessClRand ] = getClusterRandomPrediction(A, partition, numPredictions, numTargetEdges)
%  How helpful is a clustering in link prediction? One way to find out is to randomly predict links by picking endpoints randomly, while ensuring that they belong to the same cluster and see the difference.
    
    numTrainingEdges = full(sum(sum(A))/2);
    numUsers = size(A,1);
    [EdgeCount] = getClusterStatistics(A, partition)
    numClusters = numel(unique(partition));
    U = triu(A);
    prCorrectPred = 0;
    prNormalizer = sum(sum(triu(EdgeCount)));
    for i=1:numClusters
        numUsers_i = sum(partition == i);
        for j=i:numClusters
            numUsers_j = sum(partition == j);
            prCluster_ij = EdgeCount(i,j)/prNormalizer;
            prCorrectPred = prCorrectPred + prCluster_ij*(1/numUsers)*(EdgeCount(i,j)/(sum(EdgeCount(i,:))))*((1/numUsers_i) + (1/numUsers_j));
        end
    end

    numCorrectEdges = numTargetEdges*numPredictions*prCorrectPred;
    completenessClRand = numCorrectEdges/numTargetEdges;
    precisionClRand = numCorrectEdges/numPredictions;
    
    fprintf(1,'Random predictor with clusters: completeness, precision: %d, %d \n',completenessClRand , precisionClRand);
end

function [predictivePower, scoreMatrix] = boostScoresWithCluster(A, scoreMatrix,  partition)
%  An attempt to improve predictive power of social network link prediction, using node clustering knowledge.
%      Input: Matrix A which was clustered, the scoreMatrix to be altered by the clustering, the partition.
%      Output: the predictivePower fraction, and the modified scoreMatrix altered according to the following rule: For every intra-cluster-i edge, Multiply the original score by predictivePower(i).
%      For each cluster:
%      check the probability of both end-points of an edge falling in the same user-cluster. Or equivalently, check the ratio of the number of intracluster edges to the number of cross-cluster edges.
    intraClusterEdges = [];
    crossClusterEdges = [];
    predictivePower = [];
    numClusters = numel(unique(partition));
    
    [EdgeCount] = getClusterStatistics(A, partition)
    for i=1:numClusters
        fprintf(1,'Processing: Cluster %d \n', i)
        
        predictivePower(i) = EdgeCount(i,i)/ (sum(EdgeCount(i,:)) - EdgeCount(i,i));
        fprintf(1, '.. intraClusterEdges/ crossClusterEdges: = %d \n', predictivePower(i));
        
        % Alter similarity score.
        display('Altering similarity score')
        members = find(partition == i);
        scoreMatrix(members, members) = predictivePower(i) * scoreMatrix(members, members);
    end
end

end
end
