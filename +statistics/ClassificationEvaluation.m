classdef ClassificationEvaluation
methods(Static=true)

function fracMisclassified = getFracMisclassified(labelsPredicted, labels)
    assert(all(size(labelsPredicted) == size(labels)), 'Mismatch in label dimensions!');
    assert(numel(unique(labelsPredicted)) ~= numel(unique(labels)) || isempty(setxor(labelsPredicted, labels)), 'Different labels being used!');
    fracMisclassified = sum(labelsPredicted ~= labels)/ numel(labelsPredicted);
end

function fracMisclassified = getClassifierMisclassificationRate(classifier, X, labels)
    import statistics.*;
    labelsPredicted = MatrixFunctions.functionalToEachColumn(X, classifier);
    fracMisclassified = ClassificationEvaluation.getFracMisclassified(labelsPredicted(:), labels(:));
end

function avgError = getMisclassificationRateFromCrossValidation(X, labels, learner, numSplits)
    %  Input:
    %       Xs: A matrix, whose columns indicate datapoints.
    %      labels: A row vector of labels corresponding to data-points.
    %      learner: Uses a suitable portion of X and labels to return a classification function.
    %      numSplits: number of splits to be made in X and labels for doing cross-validation.
    %  Output:
    %      avgPerformance : the expected goodness.
    import randomizationUtils.*;
    import statistics.*;

    learner = @(TrainingSet)learner(TrainingSet.X, TrainingSet.y);

    tester = @(classifier, TestSet)ClassificationEvaluation.getClassifierMisclassificationRate(classifier, TestSet.X, TestSet.y);

    avgError = Estimation.getGeneralizationAbilityCrossValidation(learner, tester, numSplits, X, labels);
    % fprintf('Avg Performance: %d\n', avgPerformance);
end

function fracMisclassificationValues = getSampleComplexity(learner, numSamplesTrainingRange, samplingFn, SamplesTest, yTest, numTrials)
    import statistics.*;
    if(~exist('numTrials'))
        numTrials = 1;
    end
    fracMisclassificationValues = [];
    for(numSamplesTraining = numSamplesTrainingRange)
        fracMissclassificationEstimates = [];
        fprintf('Num samples: %d \n', numSamplesTraining);
        for trial = 1:numTrials
            [SamplesTraining, yTraining] = samplingFn(numSamplesTraining);
            classifier = learner(SamplesTraining, yTraining);
            fracMissclassificationEstimates(end+1) = ClassificationEvaluation.getClassifierMisclassificationRate(classifier, SamplesTest, yTest);
        end
        fracMisclassificationValues(end+1) = mean(fracMissclassificationEstimates);
    end
end

function [precision, sensitivity, specificity] = checkPrediction(sortedScoresAll, sortedIndicesAll, numPredictions, targetItemSet)
%          Description:
%              Calculate the expected precision, sensitivity and specificity if one were to predict (or identify as positives) numPredictions items using scores provided in sortedScoresAll.
%              This method can also be used to approximate AUC under ROC curve, truncated to just span the range 1:maxNumPredictions.
%              The logic is specified in the following:
%              'AUC from Scores' by nAgarAjan naTarAjan and vishvAs vAsuki. http://userweb.cs.utexas.edu/~vvasuki/work/statistics/AUCFromScores/AUCFromScores.pdf
%          WARNING: Minimal error checking. Please send valid inputs :-).
%          Input:
%              sortedScoresAll: vector. The sorted scores assigned to all items of interest.
%              sortedIndicesAll: vector. The indices of items corresponding to scores in sortedScoresAll.
%              numPredictions: scalar. The number of predictions to be made using the score vector.
%              targetItemSet: logical vector. ith element is 1 if the ith item is in the target set, and 0 otherwise.
%          ASSUMPTION: This assumes that items which should NEVER be predicted have score = -INF
%          Output:
%              precision: the fraction of predicted items which happen to be target items.
%              sensitivity: the fraction of target Items in the prediction.
%              specificity: the fraction of non-target items correctly excluded from the prediction.
%          Usage: refer test methods. Note: +statistics folder should be in the path.
    
%      display('Checking prediction.')
    assert(numel(targetItemSet) == numel(sortedScoresAll), 'Size mismatch between targetItemSet and sortedScoresAll.');
    targetItemSet = targetItemSet(:);
    
    [sortedScoresTrimmed, sortedIndices] = statistics.ClassificationEvaluation.trimSortedScoresItems(numPredictions, sortedScoresAll, sortedIndicesAll);
    
    cutoff = sortedScoresTrimmed(end);
    firstElementInEq = find(sortedScoresTrimmed == cutoff, 1, 'first');
    eq = sortedIndices(firstElementInEq: end);
    gt = sortedIndices(1: firstElementInEq - 1);
    
    numIdentifiedPositives = full(sum(targetItemSet(gt)));
    numIdentifiedPositives = numIdentifiedPositives + full(sum(targetItemSet(eq)))*(numPredictions - numel(gt))/numel(eq);
    
%      Called T in the document.
    numActualPositives = sum(targetItemSet);
    
%      Called U in the document.
    numCandidateItems = numel(targetItemSet) - sum(sortedScoresAll == -Inf);
    
    numActualNegatives = numCandidateItems - sum(targetItemSet);
    numMisidentifiedPositives = numPredictions - numIdentifiedPositives;
    
    precision =  numIdentifiedPositives / numPredictions;
    
    sensitivity = numIdentifiedPositives/ numActualPositives;
    specificity = 1 - numMisidentifiedPositives/numActualNegatives;
end

function [sortedScores, sortedIndices] = trimSortedScoresItems(numPredictions, sortedScores, sortedIndices)
    cutoff = sortedScores(numPredictions);
%      display('Found cutoff score')
    lastOccuranceOfCutoff = numPredictions - 1 + find(sortedScores(numPredictions: end) == cutoff, 1, 'last');
    sortedIndices = sortedIndices(1:lastOccuranceOfCutoff);
    sortedScores = sortedScores(1:lastOccuranceOfCutoff);
end

function [precisions, sensitivities, specificities] = checkPredictions(sortedScores, sortedIndices, numPredictionsVector, targetItemSet)
    precisions = zeros(size(numPredictionsVector));
    sensitivities = zeros(size(numPredictionsVector));
    specificities = zeros(size(numPredictionsVector));
    i = 1;
    for numPredictions=numPredictionsVector
        [precisions(i), sensitivities(i), specificities(i)] = statistics.ClassificationEvaluation.checkPrediction(sortedScores, sortedIndices, numPredictions, targetItemSet);
        i = i + 1;
    end
end

function [areaUnderCurve] = getAUCApprox(targetItemSet, scoreVector, granularity, maxNumPredictions)
%          Description:
%              Suppose that you want to identify n (where n varies from 1 to N) items as belonging to a target set, using scores provided in scoreVector. How good is this series of predictions? To find out, one can calculate the the expected AUC (Area under the ROC curve) approximately by using a piecewise-linear function.
%              The logic is specified in the following:
%              'AUC from Scores' by nAgarAjan naTarAjan and vishvAs vAsuki. http://userweb.cs.utexas.edu/~vvasuki/work/statistics/AUCFromScores/AUCFromScores.pdf
%          WARNINGS: 1. Minimal error checking. Please send valid inputs :-).
%                    2. Known to be inaccurate if maxNumPredictions << possible number of predictions.
%          Input:
%              targetItemSet: logical vector. ith element is 1 if the ith item is in the target set, and 0 otherwise.
%              scoreVector: vector. the ith element corresponds to score of item i.
%              maxNumPredictions(optional): scalar. Used to calculate  approximate E[AUC] under ROC curve, truncated to just span the range 1:maxNumPredictions.
%          ASSUMPTION: This assumes that items which should NEVER be predicted have score = -INF
%          Output:
%              AUC.
%          Usage: refer test methods. Note: +statistics folder should be in the path.
        
    areaUnderCurve = 0;
    areaUnderCurveUpperBound = 0;
    areaUnderCurveLowerBound = 0;
    oldSpecificity = 1;
    oldSensitivity = 0;
    
    
    [sortedScoresAll sortedIndicesAll] = sort(full(scoreVector),'descend');
    numCandidateItems = numel(targetItemSet) - sum(scoreVector == -Inf);
    
    if nargin < 4
        maxNumPredictions = numCandidateItems;
    end
    
    if maxNumPredictions > numCandidateItems
        error('maxNumPredictions exceeds the possible range of number of predictions');
    end
        
    numTargetItems = full(sum(targetItemSet));
    for numPredictions = floor(linspace(1, maxNumPredictions, min(maxNumPredictions, granularity)))
        [precision, sensitivity, specificity] = statistics.ClassificationEvaluation.checkPrediction(sortedScoresAll, sortedIndicesAll, numPredictions, targetItemSet);
%              fprintf (1, '%d %d %d \n', numPredictions, sensitivity, specificity);
        
        areaUnderCurveUpperBound = areaUnderCurveUpperBound + sensitivity*(oldSpecificity - specificity);
        areaUnderCurveLowerBound = areaUnderCurveLowerBound + oldSensitivity*(oldSpecificity - specificity);
        oldSpecificity = specificity;
        oldSensitivity = sensitivity;
    end
    areaUnderCurve = areaUnderCurveLowerBound;
    areaUnderCurve = (areaUnderCurveLowerBound + areaUnderCurveUpperBound)/2;
    
end

function [areaUnderCurve] = getAUCEstimate(targetItemSet, scoreVector)
%      Expression was proposed by David Hand and Robert Till in their Machine Learning journal paper published nine years ago, which can be found at http://www.springerlink.com/content/nn141j42838n7u21/fulltext.pdf. 
%          ASSUMPTION: This assumes that items which should NEVER be predicted have score = -INF
%          Output:
%              AUC.
%          Usage: refer test methods. Note: +statistics folder should be in the path.
    error('Implementation and testing incomplete');
    [sortedScoresAll sortedIndicesAll] = sort(full(scoreVector),'descend');
    numTargetItems = full(sum(targetItemSet));
    numCandidateItems = numel(targetItemSet) - sum(scoreVector == -Inf);
    numNegativeItems = numCandidateItems - numTargetItems;
    sumTargetRanks = sum(find(targetItemSet(sortedIndicesAll)));
    sumNegativeRanks = numCandidateItems*(numCandidateItems + 1)/2 - sumTargetRanks;
    areaUnderCurve = (sumNegativeRanks - numNegativeItems*(numNegativeItems+1)/2)/(numNegativeItems*numTargetItems);
    

%      Wei Does this:
    areaUnderCurve = ((numNegativeItems*numTargetItems) - (sumTargetRanks - numTargetItems*(numTargetItems + 1)/2))/ (numNegativeItems*numTargetItems);

%       Wei's implementation is below, for reference/ comparison.
%          [dummy,sidx] = sort(scores(testIdx),'descend');
%          nPos = sum(labels==true); nNeg = sum(labels==false);
%          auc(1,j) = 1-(sum(find(labels(sidx)))-nPos*(nPos+1)/2)/(nPos*nNeg);
end

function [areaUnderCurve] = getAUC(targetItemSet, scoreVector, maxNumPredictions)
%          Description:
%              Suppose that you want to identify n (where n varies from 1 to N) items as belonging to a target set, using scores provided in scoreVector. How good is this series of predictions? To find out, one can calculate the the expected AUC (Area under the ROC curve) exactly. 
%              This method can also be used to calculate AUC under ROC curve, truncated to just span the range 1:maxNumPredictions.
%              The logic is specified in the following:
%              'AUC from Scores' by nAgarAjan naTarAjan and vishvAs vAsuki. http://userweb.cs.utexas.edu/~vvasuki/work/statistics/AUCFromScores/AUCFromScores.pdf
%          WARNING: Minimal error checking. Please send valid inputs :-).
%          Input:
%              targetItemSet: logical vector. ith element is 1 if the ith item is in the target set, and 0 otherwise.
%              scoreVector: vector. the ith element corresponds to score of item i.
%              maxNumPredictions(optional): scalar. Used to calculate E[AUC] under ROC curve, truncated to just span the range 1:maxNumPredictions.
%          ASSUMPTION: This assumes that items which should NEVER be predicted have score = -INF
%          Output:
%              AUC.
%          Usage: refer test methods. Note: +statistics folder should be in the path. Eg: statistics.ClassificationEvaluation.getAUC(.,.)
    error('Implementation and testing incomplete');
    
    targetItemSet = logical(targetItemSet);
    uniqSortedTgtScores = unique(scoreVector(targetItemSet));
    numCandidateItems = numel(targetItemSet) - sum(scoreVector == -Inf);
    
    if nargin < 3
        maxNumPredictions = numCandidateItems;
    end
    
    if maxNumPredictions > numCandidateItems
        error('maxNumPredictions exceeds the possible range of number of predictions');
    end
    
    numTargetItems = sum(targetItemSet);
    areaUnderCurve = 0;
    widthCovered = sum(scoreVector > uniqSortedTgtScores(end));
    if widthCovered >= maxNumPredictions
        return;
    end
    
    bMaxPredictionsDone = false;
    for i=numel(uniqSortedTgtScores):-1:1
        score = uniqSortedTgtScores(i);
        
%              Interval of type A
        I_U = find(scoreVector == score);
        width = length(I_U);
        if widthCovered + width >= maxNumPredictions
            width = maxNumPredictions - widthCovered;
            bMaxPredictionsDone = true;
        end
        widthCovered = widthCovered + width;
            
        numI_T = sum(targetItemSet(I_U));
        sensitivity_I = sum(scoreVector(targetItemSet) > score)/numTargetItems;
        expectedCorrectIdentifications = numI_T*width*(width+1)/(2*length(I_U));
        changeInAreaUnderCurve = sensitivity_I * width / numCandidateItems + (expectedCorrectIdentifications)/(numTargetItems*numCandidateItems);
        areaUnderCurve = areaUnderCurve + changeInAreaUnderCurve;
        if bMaxPredictionsDone
            break;
        end
            
%              keyboard
%              fprintf(1, 'AUC so far: %d\n', areaUnderCurve);
            
%              Interval of type B
        if i == 1
            nextScore = -Inf;
        else
            nextScore = uniqSortedTgtScores(i-1);
        end
        I_U = find((scoreVector < score) & (scoreVector > nextScore));
        width = length(I_U);
        if widthCovered + width >= maxNumPredictions
            width = maxNumPredictions - widthCovered;
            bMaxPredictionsDone = true;
        end
        widthCovered = widthCovered + width;
        sensitivity_I = sum(scoreVector(targetItemSet) > nextScore)/numTargetItems;
%              keyboard
        changeInAreaUnderCurve = sensitivity_I * width / numCandidateItems;
        areaUnderCurve = areaUnderCurve + changeInAreaUnderCurve;
%              fprintf(1, 'AUC so far: %d\n', areaUnderCurve);
        if bMaxPredictionsDone
            break;
        end
    end
end


function testGetAUC()
    numItems = 1000;
    scoreVector = zeros(numItems, 1);
    targetItemSet = zeros(numItems, 1);
    targetItemSet(5) = 1;
    granularity = Inf;
    
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCApprox(targetItemSet, scoreVector, granularity)
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCEstimate(targetItemSet, scoreVector)
    fprintf(1, 'expected AUC: %d\n', 1/2);
    
    scoreVector(5) = 1;
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCApprox(targetItemSet, scoreVector, granularity)
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCEstimate(targetItemSet, scoreVector)
    fprintf(1, 'expected AUC: %d\n', 1);
    
    scoreVector(4) = 1;
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCApprox(targetItemSet, scoreVector, granularity)
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCEstimate(targetItemSet, scoreVector)
    fprintf(1, 'expected AUC: close to %d\n', 1);

    scoreVector(9) = 1;
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCApprox(targetItemSet, scoreVector, granularity)
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCEstimate(targetItemSet, scoreVector)
    fprintf(1, 'expected AUC: close to %d\n', 1);

    scoreVector(9:508) = 1;
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCApprox(targetItemSet, scoreVector, granularity)
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCEstimate(targetItemSet, scoreVector)
    fprintf(1, 'expected AUC: close to %d\n', 0.75);
    
end

function testGetAUCTruncated()
    numItems = 1000;
    granularity = 50;
    scoreVector = zeros(numItems, 1);
    targetItemSet = zeros(numItems, 1);
    targetItemSet(5) = 1;
    
    scoreVector(5) = 1;
    maxNumPredictions = 1;
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCApprox(targetItemSet, scoreVector, granularity, maxNumPredictions)
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUC(targetItemSet, scoreVector, maxNumPredictions)
    fprintf(1, 'expected AUC: close to %d\n', 1/(numItems*sum(targetItemSet)));

    scoreVector(4) = 2;
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCApprox(targetItemSet, scoreVector, granularity, maxNumPredictions)
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUC(targetItemSet, scoreVector, maxNumPredictions)
    fprintf(1, 'expected AUC: close to %d\n', 0);

    maxNumPredictions = 2;
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCApprox(targetItemSet, scoreVector, granularity, maxNumPredictions)
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUC(targetItemSet, scoreVector, maxNumPredictions)
    fprintf(1, 'expected AUC: close to %d\n', 1/(numItems*sum(targetItemSet)));

    targetItemSet(6) = 1;
    scoreVector(6) = 1;
    maxNumPredictions = 3;
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCApprox(targetItemSet, scoreVector, granularity, maxNumPredictions)
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUC(targetItemSet, scoreVector, maxNumPredictions)
    fprintf(1, 'expected AUC: close to %d\n', 3/(numItems*sum(targetItemSet)));

    maxNumPredictions = 10;
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUCApprox(targetItemSet, scoreVector, granularity, maxNumPredictions)
    [areaUnderCurve] = statistics.ClassificationEvaluation.getAUC(targetItemSet, scoreVector, maxNumPredictions)
    fprintf(1, 'expected AUC: close to %d\n', 1);

end

function [avgPrecisions, avgSensitivities, avgSpecificities] = getAveragePerformance(ScoreMatrix, ItemsToIgnore, TargetEdgeSet, numPredictionsVector)
%  Input: ScoreMatrix, ItemsToIgnore, TargetEdgeSet, numPredictionsVector
%  Output: avgPrecisions, avgSensitivities, avgSpecificities : averaged along rows.
%  Tip: To average over columns, transpose all inputs!
%  We eliminate all zero - rows in TargetEdgeSet while measuring average performance!
    [numRows, numCols] = size(ItemsToIgnore);
    avgPrecisions = zeros(size(numPredictionsVector));
    avgSensitivities = zeros(size(numPredictionsVector));
    avgSpecificities = zeros(size(numPredictionsVector));
    currTime = cputime;
    
    rowsForPrediction = find(sum(TargetEdgeSet, 2) > 0);
    numRowsForPrediction = numel(rowsForPrediction);
    for rowNum = 1:numRowsForPrediction
        rowId = rowsForPrediction(rowNum);
        if ~mod(rowId,200)
            fprintf('Entity %d\n', rowId);
        end
        scoreVector = ScoreMatrix(rowId, :);

        [sortedScores, sortedIndices] = statistics.Ordering.processSimilarityMatrix(ItemsToIgnore(rowId, :)', scoreVector);
        
        [precisions, sensitivities, specificities] = statistics.ClassificationEvaluation.checkPredictions(sortedScores, sortedIndices, numPredictionsVector, TargetEdgeSet(rowId, :));
        
        avgPrecisions = avgPrecisions + precisions;
        avgSensitivities = avgSensitivities + sensitivities;
        avgSpecificities = avgSpecificities + specificities;
    end
    fprintf(1,'Time elapsed: %d\n', cputime - currTime);
    
    avgPrecisions = avgPrecisions/numRowsForPrediction;
    avgSensitivities = avgSensitivities/numRowsForPrediction;
    avgSpecificities = avgSpecificities/numRowsForPrediction;
end

function testCheckPrediction()
    scoreVector = zeros(10,1);
    sortedIndices = (1:10);
    targetItemSet = zeros(10,1);
    targetItemSet([2, 3, 7]) = 1;
    numPredictions = 3;
    
    [precision, sensitivity, specificity] = statistics.ClassificationEvaluation.checkPrediction(scoreVector, sortedIndices, numPredictions, targetItemSet)
    fprintf(1, 'expected result: sensitivity: %d specificity: %d\n', numPredictions/ 10, 1 - numPredictions/ 10);
end

function test_getAvgGeneralizationAbilityClassification
    import statistics.*;
    X = (rand(2, 300)>0.5);
    labels = X(1, :);
    X(1, 250:end) = ~X(1, 250:end);
    
    classifier = @(x, TrainingX, trainingLabels)x(2);
    trainingXs = {X};
    trainingLabels = {labels};
    testXs = {X};
    testLabels = {labels};
    generalizationAbility = ClassificationEvaluation.getAvgGeneralizationAbilityClassification(trainingXs, trainingLabels, testXs, testLabels, learner);
end

function testClass
    display 'Class definition is ok';
end

end
end