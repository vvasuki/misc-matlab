classdef Agglomerative
methods(Static=true)
    function partition = singleLink(X, numClasses)
        numPoints = size(X, 1);
        partition = 1:numPoints;
        numPartitions = numel(unique(partition));
        while numPartitions > numClasses
%              fprintf(1, '%d ', partition');
%              display ''
%              Calculate distances
            InterPointDistance = ClusterUtils.getInterPointDistance(X);
            InterClusterDistance = ClusterUtils.getInterClusterDistanceMin(InterPointDistance, partition);

%              Identify clusters to merge
%              Fix InterClusterDistance to have large diagonal elements.
            InterClusterDistance = InterClusterDistance + diag(sum(InterClusterDistance));
            [minValue, linIndex] = min(InterClusterDistance(:));
            [cluster_i, cluster_j] = ind2sub([numPartitions, numPartitions], linIndex);
            
%              Merge clusters.
            classes = unique(partition);
            clusterLabel_i = classes(cluster_i);
            clusterLabel_j = classes(cluster_j);
            clusterMembers_i = find(partition==clusterLabel_i);
            partition(clusterMembers_i) = clusterLabel_j;
%              fprintf(1, 'Merged Clusters %d and %d \n', clusterLabel_i, clusterLabel_j);
            
            numPartitions = numel(unique(partition));
        end
        partition = ClusterUtils.renameClusters(partition);
    end

end
end
