classdef ClusterUtils
methods(Static = true)
function [EdgeCount] = getClusterStatistics(A, partition)
    %  Output: EdgeCount: A matrix whose i,j element represents the number of edges between nodes in cluster i and nodes in cluster j.
    numClusters = numel(unique(partition));
    U = triu(A);
    numEdges = full(sum(sum(U)));
    EdgeCount = zeros(numClusters, numClusters);
    for i=1:numClusters
        for j=1:numClusters
            nodesIn_i = find(partition == i);
            nodesIn_j = find(partition == j);
            EdgeCount(i,j) = sum(sum(U(nodesIn_i, nodesIn_j)));
        end
        fprintf(1, 'Intra-partition edges %d: %d: A %d fraction.\n', i, EdgeCount(i,i), EdgeCount(i,i)/ sum(EdgeCount(i,:)));
    end
    intraClusterEdges = sum(diag(EdgeCount));
    fprintf(1, 'Avg Intra-partition edges: %d: A %d fraction.\n', intraClusterEdges, intraClusterEdges/ numEdges);
end

end
end