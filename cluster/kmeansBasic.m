function [partition] = kmeans(X, numClasses)
    %  Get random paritioning
    numPoints = size(X, 1);
    partition = getRandomPartitioning(numPoints, numClasses);
    obj = getObjective(X, partition);
    improvement = 1;
    
    while improvement > .001
        partition = repartition(X, partition);
        newObj = getObjective(X, partition);
        improvement = obj - newObj;
        obj = newObj;
%          fprintf(1,'Obj, improvement: %d %d\n', obj, improvement); 
        
    end
    partition = ClusterUtils.renameClusters(partition);
end

function partition = repartition(X, partition)
    numClasses = numel(unique(partition));
    numPoints = size(X, 1);
    DistancesFromMean = zeros(numPoints, numClasses);
    for c=1:numClasses
        clusterMembers = find(partition == c);
        numClusterMembers = numel(clusterMembers);
        clusterMean = sum(X(clusterMembers,:), 1)/numClusterMembers ;
        for i=1:numPoints
            DistancesFromMean(i, c) = Metrics.klDivergence(X(i,:), clusterMean);
        end
    end
    [y,partition] = min(DistancesFromMean,[],2);
end

function obj = getObjective(X, partition)
    numClasses = numel(unique(partition));
    numPoints = size(X, 1);
    obj = 0;
    for c=1:numClasses
        clusterMembers = find(partition == c);
        numClusterMembers = numel(clusterMembers);
        clusterMean = sum(X(clusterMembers,:), 1)/numClusterMembers ;
%          fprintf(1,'Cluster: %d\n', c); 
        for i=1:numClusterMembers 
            obj = obj + Metrics.klDivergence(X(i,:), clusterMean);
%              fprintf(1,'  Update: %d %d\n', Metrics.klDivergence(X(i,:), clusterMean), obj)
        end
    end
end

function partition = getRandomPartitioning(numPoints, numClasses)
    partition = zeros(numPoints, 1);
    for i=1:numClasses
        unPartitionedPts = find(partition == 0);
        selectedPts = randomizationUtils.Sample.sampleWithoutReplacement(unPartitionedPts, numPoints/numClasses);
        partition(selectedPts) = i;
    end
end