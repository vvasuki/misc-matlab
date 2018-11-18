classdef Estimation
methods(Static=true)
function CooccuranceProbabilities = estimateCooccuranceProbabilities(Sample)
    [dim, numSamples] = size(Sample);
    Sample = double(Sample);
    assert((numel(unique(Sample)) <= 2), 'Samples should be indicator vectors!');
    CooccuranceProbabilities = zeros(dim);
    for(sampleNum = 1:numSamples)
        samplePt = Sample(:, sampleNum);
        CooccuranceProbabilities = CooccuranceProbabilities + samplePt*samplePt';
    end
    CooccuranceProbabilities = CooccuranceProbabilities /numSamples;
end

function MutualInfo = getMutualInformationMatrix(Sample)
    [dim, numSamples] = size(Sample);
    Sample = double(Sample);
    MutualInfo = zeros(dim);
    for(i=1:dim)
        for(j=1:i)
            MutualInfo(i, j) = mutualinfo(Sample(i, :), Sample(j, :));
            MutualInfo(j, i) = MutualInfo(i, j);
        end
    end
end

function [testSets, trainingSets] = getExampleSetsForCrossValidation(numSplits, X, y)
    testSets = {};
    trainingSets = {};
    numPoints = size(X, 2);
    
    function [ZTest, ZTraining] = splitPoints(selectedExamples, Z)
        ZTest = Z(:, selectedExamples);
        trainingExampleIndices = logical(ones(1, numPoints));
        trainingExampleIndices(selectedExamples) = false;
        ZTraining = Z(:, trainingExampleIndices);
    end

    
    indexSetsTmp = VectorFunctions.getRoughlyEqualParts(1:numPoints, numSplits);
    shuffledIndices = randperm(numPoints);
    for i = 1:numSplits
        selectedExamples = shuffledIndices(indexSetsTmp{i});
        [XTest, XTraining] = splitPoints(selectedExamples, X);
        if(exist('y', 'var'))
            [yTest, yTraining] = splitPoints(selectedExamples, y);
            trainingSample.X = XTraining;
            trainingSample.y = yTraining;
            testSample.X = XTest;
            testSample.y = yTest;

            bCheck = (size(testSample.X, 2) +  size(trainingSample.X, 2) == size(X, 2));
            % fprintf(' Testing: size(testSample.X, 2) +  size(trainingSample.X, 2) == size(X, 2): %d\n', bCheck);
        else
            trainingSample = XTraining;
            testSample = XTest;
        end
        testSets{end+1} = testSample;
        trainingSets{end+1} = trainingSample;
    end
end

function avgPerformance = getGeneralizationAbilityCrossValidation(learner, tester, numSplits, X, y)
    import statistics.*;
    if(exist('y', 'var'))
        [testSets, trainingSets] = Estimation.getExampleSetsForCrossValidation(numSplits, X, y);
    else
        [testSets, trainingSets] = Estimation.getExampleSetsForCrossValidation(numSplits, X);
    end
    
    avgPerformance = Estimation.getAvgGeneralizationAbility(trainingSets, testSets, learner, tester);
end

function avgPerformance = getAvgGeneralizationAbility(trainingSets, testSets, learner, tester)
    %  Input: trainingSets: cell array. May include dataPoints and labels.
    %      testSets: cell array, of same size as trainingSets.
    %      learner: Uses a member of trainingSets to return a classification function.
    %      tester: Inputs a classification function and a member of testSets and returns some measure of performance.
    %  Output: avgPerformance
    assert(length(trainingSets) == length(testSets), 'Error: mismatch in training and test set sizes.');

    numTrainingSets = length(trainingSets);
    performanceMeasures = [];
    sampleSizes = [];
    for(i = 1:numTrainingSets)
        trainingSet = trainingSets{i};
        performanceMeasures(end+1) = tester(learner(trainingSet), testSets{i});
        if(isstruct('trainingSet'))
            sampleSizes(end+1) = size(trainingSet.X, 2);
        else
            sampleSizes(end+1) = size(trainingSet, 2);
        end
    end
    sampleSizes = sampleSizes/sum(sampleSizes);
    avgPerformance = sum(performanceMeasures.*sampleSizes);
end

function test_getExampleSetsForCrossValidation()
    import statistics.*;
    numSplits = 3;
    y = 1:10;
    [testSets, trainingSets] = Estimation.getExampleSetsForCrossValidation(numSplits, y);
    for i=1:numSplits
        testSets{i}
        % trainingSets{i}
    end
end

function testClass
    display 'Class definition is ok';
end

end
end