classdef ClusterUtils
%  For graph clustering utilities, see elsewhere.

methods(Static=true)
    function [confusionMatrix] = getConfusionMatrix(partition1, partition2)
        numClasses = numel(unique(partition1));
        confusionMatrix = zeros(numClasses, numClasses);
        for i=1:numClasses
            for j=1:numClasses
                confusionMatrix(i, j) = sum((partition1 == i).*(partition2 == j));
            end
        end
        display 'Columns may need to be transposed appropriately.';
    end
    
    function InterPointDistance = getInterPointDistance(X)
        numPoints = size(X, 1);
        InterPointDistance = zeros(numPoints,numPoints);
        for i = 1:numPoints
        for j = 1:numPoints
            InterPointDistance(i, j) = Metrics.klDivergence(X(i,:), X(j,:));
        end
        end
    end

    function InterClusterDistance = getInterClusterDistanceMin(InterPointDistance, partition)
        numPoints = size(InterPointDistance, 1);
        classes = unique(partition);
        numClasses=numel(classes);
        InterClusterDistance = zeros(numClasses,numClasses);
        for i = 1:numClasses
            clusterLabel_i = classes(i);
            clusterMembers_i = find(partition == clusterLabel_i);
            for j = 1:numClasses
                clusterLabel_j = classes(j);
                clusterMembers_j = find(partition == clusterLabel_j);
%                  fprintf(1, 'InterClusterDistance: comparing: %d %d %d %d \n', i, j, clusterLabel_i, clusterLabel_j);
                InterClusterDistance(i, j) = min(min(InterPointDistance(clusterMembers_i, clusterMembers_j)));
            end
        end
    end
    
    function [partition] = renameClusters(oldPartition)
        classes = unique(oldPartition);
        numClasses=numel(classes);
        partition = oldPartition;
        for i = 1:numClasses
            clusterLabel_i = classes(i);
            clusterMembers_i = find(oldPartition == clusterLabel_i);
            partition(clusterMembers_i) = i;
        end
    end

end
end