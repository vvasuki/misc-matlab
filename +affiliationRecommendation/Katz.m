classdef Katz
methods(Static=true)
function ScoreMatrix = getKatzApprox(UserUser, Training, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams)
    numIterations = 3;
%      display 'Learning parameters';
%      currTime = cputime;
%      bestBeta = affiliationRecommendation.Katz.tKatzParameterLearner(numIterations, TrainingValid, ValidationEdgeSet(:), numPredictionsForValidation);
%      fprintf(1,'Time elapsed: %d\n', cputime - currTime);

%      bestBeta was found to be 10^-12 for Orkut and Youtube.
    bestBeta = 10^-12
    display 'Computing score matrix';
    currTime = cputime;
    ScoreMatrix = affiliationRecommendation.Katz.getTKatzMatrix(Training, numIterations, bestBeta);
    fprintf(1,'Time elapsed: %d\n', cputime - currTime);
end

function ScoreMatrix = getTKatzMatrix(TrainingValid, numIterations, beta)
    display('Calculating Truncated Katz')
    ScoreMatrix = linkPrediction.Katz.KatzApprox(TrainingValid*TrainingValid', numIterations, beta);
    ScoreMatrix = ScoreMatrix * TrainingValid;
    display('Got Katz matrix')
end

function [bestBeta] = tKatzParameterLearner(numIterations, TrainingValid, validationEdgeSet, numPredictions)
    
    FoldedTrainingUG = full(TrainingValid*TrainingValid');
    precisionKatzOld = 0;
    bestPrecision = 0;
    bestBeta = 0;
    powersFoldedTrainingUG = cell(numIterations,1);
    powersFoldedTrainingUG{1} = FoldedTrainingUG;
    display 'Computing powers for training validation.'
    for i=2:numIterations
        powersFoldedTrainingUG{i} = FoldedTrainingUG*powersFoldedTrainingUG{i-1};
    end
    display 'Computed powers for training validation.'
    for betaTestPow=-12:1:-7
        betaTest = 10^betaTestPow;
        ScoreMatrix = linkPrediction.Katz.getKatzMatrixFromPowers(powersFoldedTrainingUG, betaTest);
        ScoreMatrix = ScoreMatrix * TrainingValid;
        
        display('Got Katz matrix')
        
        [scoreVector, sortedIndices] = statistics.Ordering.processSimilarityMatrix(TrainingValid, ScoreMatrix);
        
        [precisionKatz, completenessKatz] = statistics.ClassificationEvaluation.checkPrediction(scoreVector, sortedIndices, numPredictions, validationEdgeSet);
        
        if(precisionKatz < precisionKatzOld)
            break;
        elseif(precisionKatz> bestPrecision)
            bestPrecision = precisionKatz;
            bestBeta = betaTest;
        end
        fprintf(1, 'beta : %d precisionKatz: %d \n', betaTest, precisionKatz);
        precisionKatzOld = precisionKatz;
    end
    fprintf(1, 'best beta is %d \n',bestBeta);
end

function precision = get3KatzPrecisionFromPowers(TrainingValid, BA, B2A, A3, validationEdgeSet, l, betaExp, numPredictions)
    ScoreMatrix = l*BA + (10^betaExp)*(l*l*B2A + A3);
    [scoreVector, sortedIndices] = statistics.Ordering.processSimilarityMatrix(TrainingValid, ScoreMatrix);
    [precision, completeness] = statistics.ClassificationEvaluation.checkPrediction(scoreVector, sortedIndices, numPredictions, validationEdgeSet);
end

function [betaExpBest, lBest] = katzJoint3ApproxParameterLearner(UserUser, TrainingValid, validationEdgeSet, numPredictions)
    import affiliationRecommendation.*;
    lBest = 0.2;
    betaExpBest = -1;
    precisionBest = 0;
    precisionOldBeta = 0;
    %  Compute the powers
    display 'Computing powers for training validation.';
    UserUser = double(UserUser);
    BA = full(UserUser*TrainingValid);
    B2A = full(UserUser*BA);
    A3 = full(TrainingValid*TrainingValid'*TrainingValid);
    display 'Computed powers for training validation.';
    
    
    objFn = @(params)-Katz.get3KatzPrecisionFromPowers(TrainingValid, BA, B2A, A3, validationEdgeSet, params(2), params(1), numPredictions);
    
    stepL = 0.2;
    lRange = 0:stepL:3.0;
    betaExpRange = -2:1:0;
    domainSets = {betaExpRange, lRange};
    [objMin, paramsBest] = optimization.DescentMethods.discreteSequentialMinimization(domainSets, objFn);
    betaExpBest = paramsBest(1);
    lBest = paramsBest(2);
    fprintf('betaExpBest: %d lBest: %d \n', betaExpBest, lBest);
end

function ScoreMatrix = getKatzJoint3Approx(UserUser, Training, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams)
    bLearnParameters = 0;
    if(bLearnParameters)
        display 'Learning parameters';
        currTime = cputime; 
        [betaExpBest, lBest] = affiliationRecommendation.Katz.katzJoint3ApproxParameterLearner(UserUser, TrainingValid, ValidationEdgeSet(:), numPredictionsForValidation);
        fprintf(1,'Time elapsed: %d\n', cputime - currTime);
    else
        if(StringUtilities.isSubstring(initParams.datasetName, 'youtube'))
            % Youtube case.
            betaExpBest = -1;
            lBest = .4;
        else
            % Orkut case.
            betaExpBest = -2;
            lBest = .2;
        end
    end
    
    display 'Computing score matrix';
    currTime = cputime;
    BA = full(UserUser*Training);
    B2A = full(UserUser*BA);
    A3 = full(Training*Training'*Training);
    ScoreMatrix = lBest*BA + (10^betaExpBest)*(lBest*lBest*B2A + A3);
    fprintf(1,'Time elapsed: %d\n', cputime - currTime);
end

function ScoreMatrix = getKatzJoint3ApproxFminunc(UserUser, Training, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams)
    display 'Learning parameters using fminunc';
    currTime = cputime; 
    [betaExpBest, lBest] = affiliationRecommendation.Katz.katzJoint3ApproxParameterLearnerFminunc(UserUser, TrainingValid, ValidationEdgeSet(:), numPredictionsForValidation);
    fprintf(1,'Time elapsed: %d\n', cputime - currTime);
    
%      betaExpBest = -1; lBest = 4.000000e-01; % for youtube
    
    display 'Computing score matrix';
    currTime = cputime;
    BA = full(UserUser*Training);
    B2A = full(UserUser*BA);
    A3 = full(Training*Training'*Training);
    ScoreMatrix = lBest*BA + (10^betaExpBest)*(lBest*lBest*B2A + A3);
    fprintf(1,'Time elapsed: %d\n', cputime - currTime);
end

function [betaExpBest, lBest] = katzJoint3ApproxParameterLearnerFminunc(UserUser, TrainingValid, validationEdgeSet, numPredictions)
    lBest = 0.2;
    betaExpBest = -1;
    precisionBest = 0;
    precisionOldBeta = 0;
    %  Compute the powers
    display 'Computing powers for training validation.';
    UserUser = double(UserUser);
    BA = full(UserUser*TrainingValid);
    B2A = full(UserUser*BA);
    A3 = full(TrainingValid*TrainingValid'*TrainingValid);
    display 'Computed powers for training validation.';
    
    Domain_beta = [-2 0];
    Domain_l = [0 2];
    lowerBound = min([Domain_beta(:)'; Domain_l(:)'], [], 2);
    upperBound = max([Domain_beta(:)'; Domain_l(:)'], [], 2);
    domains = {Domain_beta, Domain_l};
        
    f_0 = @(params) affiliationRecommendation.Katz.get3KatzPrecisionFromPowers(TrainingValid, BA, B2A, A3, validationEdgeSet, params(2), params(1), numPredictions);
    
    extendedValueFnHandle = @(x)functionals.Functionals.extendedValueFunctionalWrapper(f_0, upperBound, lowerBound, x);

    x_start = lowerBound;
    options = optimset('Display','iter','TolFun',1e-2);
    options = optimset(options,'TolX',1e-2);
    display 'Calling optimization subroutine...';
    bestParameters = fmincon(extendedValueFnHandle, x_start, [], [], [], [], lowerBound, upperBound, [], options);
    betaExpBest = bestParameters(1);
    lBest = bestParameters(2);
%      keyboard
end

function testClass()
    display 'Class is ok!'
end

end
end