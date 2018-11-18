classdef SVDLatentFactors
methods(Static=true)
function [numFactorsBest, lBest] = learnParametersSeqMin(UserUser, Training, validationEdgeSet, mergedNetType, domainSets, numPredictions)
    import affiliationRecommendation.*;
    import optimization.*;
    fprintf('In learnParametersSeqMin: numPredictions: %d\n', numPredictions);
    scoreGenerator = @(parameters)SVDLatentFactors.getScoreMatrix(UserUser, Training, parameters(1), parameters(2), mergedNetType);
    objFn = @(params)-statistics.Ordering.getPerformance(scoreGenerator, Training, validationEdgeSet, params, numPredictions);
    
    [objMin, paramsBest] = DescentMethods.discreteSequentialMinimization(domainSets, objFn);
    numFactorsBest = paramsBest(1);
    lBest = paramsBest(2);
    fprintf(1, 'Best on validation set: numFactors: %d, l: %f\n', numFactorsBest, lBest);
end

function MergedNet = getMergedNet(UserUser, UGTraining, l, mergedNetType)
    [numUsers, numGroups] = size(UGTraining);
    if(mergedNetType == 6)
        D = UGTraining'*UserUser*UGTraining;
        MergedNet = [l*UserUser UGTraining; UGTraining' l*D/max(max(D))];
    elseif(mergedNetType == 5)
        D = UGTraining'*UGTraining;
        MergedNet = [l*UserUser UGTraining; UGTraining' l*D/max(max(D))];
    elseif(mergedNetType == 4)
        D = UGTraining'*UGTraining;
        MergedNet = [l*UserUser UGTraining; UGTraining' D/max(max(D))];
    elseif(mergedNetType == 3)
        D = UGTraining'*UGTraining;
        MergedNet = [l*UserUser UGTraining; UGTraining' l*D/norm(UGTraining, 'fro')];
    elseif(mergedNetType == 2)
        D = UGTraining'*UGTraining;
        MergedNet = [l*UserUser UGTraining; UGTraining' D/norm(UGTraining, 'fro')];
    elseif(mergedNetType == 1)
        MergedNet = [l*UserUser UGTraining; UGTraining' sparse(numGroups, numGroups)];
    else
        MergedNet = [UGTraining l*UserUser]; 
    end
    fprintf(1,'Merged net created\n');

end

function [UU,VV,DD,FF] = combinedLowRank(UserUser,k1,UserGroup,k2)
    % Returns low rank approximation of UserUser and UserGroup:
    % UserUser approx UU*DD*UU'
    % UserGroup approx UU*FF*VV'
    % Rank of both approximations is k1+k2
    % [UU,VV,DD,FF] = combinedLowRank(UserUser,k1,UserGroup,k2)
    
    % compute low rank approximation of UserUser
    UserUser = double(UserUser);
    eigsOpt.disp = 0;
    [U,D] = eigs(UserUser, k1, 'LM', eigsOpt);
    
    % compute low rank approximation of UserGroup
    [V,F,W] = svds(UserGroup,k2);
    
    % Common subspacee
    [UU,F] = qr([U V],0);
    
    % Compute middel factor in approximation of UserUser
    DD = UU'*UserUser*UU;
    
    % Compute right subspace for UserGroup
    V = UU'*UserGroup;
    [U,F,VV] = svd(V,'econ');
    
    % compute middle factor in approximation of UserGroup
    FF = V*VV;
end




function [Factors, S] = mergeAndApproximate(UserUser, UGTraining, numFactors, l, mergedNetType)
    import affiliationRecommendation.*;
    MergedNet = SVDLatentFactors.getMergedNet(UserUser, UGTraining, l, mergedNetType);
    currTime = cputime;
    % [UserFactors,S,GroupFactors] = svds(MergedNet, numFactors);
    OPTS.issym = true;
    OPTS.isreal = true;
    OPTS.disp = 0;
    [Factors, S] = eigs(MergedNet, numFactors, 'LM', OPTS);
    fprintf(1,'Time elapsed: %d\n', cputime - currTime);
    
end

function ScoreMatrix = getScoreMatrix(UserUser, UGTraining, numFactors, l, mergedNetType, usersToEvaluate)
    import affiliationRecommendation.*;
    [numUsers, numGroups] = size(UGTraining);
    if ~exist('usersToEvaluate')
        usersToEvaluate = 1:numUsers;
    end
    [Factors, S] = SVDLatentFactors.mergeAndApproximate(UserUser, UGTraining, numFactors, l, mergedNetType);
%          scoreMatrixSVD = UserFactors*S*GroupFactors';
%          Storing scoreMatrixSVD, as calculated above, is too costly. so must do something else.
%          keyboard
    ScoreMatrix = Factors(usersToEvaluate, :)*S*(Factors(numUsers+1:end, :)');
end

function ScoreMatrix = getScoreMatrixSVD(UGTraining, numFactors, usersToEvaluate)
    [numUsers, numGroups] = size(UGTraining);
    if ~exist('usersToEvaluate')
        usersToEvaluate = 1:numUsers;
    end
    [UserFactors,S,GroupFactors] = svds(UGTraining, numFactors);
    UserFactors = UserFactors(usersToEvaluate, :) * S;
    ScoreMatrix = UserFactors*GroupFactors';
end

function numFactorsBest = svdApproxParameterLearner(UGTraining, validationEdgeSet, numPredictions)
%  Does SVD(Affiliation network), learns the best numFactors to avoid overfitting.
    import affiliationRecommendation.*;
    import statistics.*;
    numFactorsBest = 1;
    precisionSVDBest = 0;
    precisionSVDOld = 0;
    numFactorsRange = 10:10:100;
    for numFactors= numFactorsRange
        ScoreMatrix = SVDLatentFactors.getScoreMatrixSVD(UGTraining, numFactors);
        [scoreVector, sortedIndices] = Ordering.processSimilarityMatrix(UGTraining, ScoreMatrix);

        [precisionSVD, completenessSVD, specificity] = statistics.ClassificationEvaluation.checkPrediction(scoreVector, sortedIndices, numPredictions, validationEdgeSet);

%          fprintf(1, 'on validation set: numFactors: %d, precisionSVD: %f\n', numFactors, precisionSVD);
        if(precisionSVD < precisionSVDOld)
            break;
        elseif(precisionSVDBest < precisionSVD)
            numFactorsBest = numFactors;
            precisionSVDBest = precisionSVD;
        end
        precisionSVDOld = precisionSVD;
    end 
end

function ScoreMatrix = getSVDApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    currTime = cputime;
    numFactorsBest = SVDLatentFactors.svdApproxParameterLearner(TrainingValid, ValidationEdgeSet(:), numPredictionsForValidation);
    fprintf(1,'Time elapsed: %d\n', cputime - currTime);
    currTime = cputime;
    ScoreMatrix = SVDLatentFactors.getScoreMatrixSVD(UGTraining, numFactorsBest, usersToEvaluate);
    fprintf(1,'Time elapsed: %d\n', cputime - currTime);
end

function ScoreMatrix = getJointSVDApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, mergedNetType, filePrefix, usersToEvaluate, bValidate, initParams)
    import affiliationRecommendation.*;
    if(~exist('bValidate') || bValidate)
        
        display 'Doing validation.'
        currTime = cputime;
        %      [numFactorsBest, lBest] = SVDLatentFactors.learnParameters(UserUser, TrainingValid, ValidationEdgeSet(:), mergedNetType, numPredictionsForValidation, filePrefix);
        stepNumFactors = 10;
        stepL = 0.2;
        numFactorsRange = 80:stepNumFactors:100;
        lRange = 0:stepL:3.0;
        domainSets = {numFactorsRange, lRange};
        [numFactorsBest, lBest] = SVDLatentFactors.learnParametersSeqMin(UserUser, TrainingValid, ValidationEdgeSet(:), mergedNetType, domainSets, numPredictionsForValidation);
        fprintf(1,'Time elapsed: %d\n', cputime - currTime);
    else
        if(StringUtilities.isSubstring(initParams.datasetName, 'youtube'))
            % Youtube case.
            numFactorsBest = 90;
            lBest = 1;
        else
            % Orkut case.
            numFactorsBest = 50;
            lBest = .8;
        end
    end
    display 'Using best parameters.'
    currTime = cputime;
    ScoreMatrix = SVDLatentFactors.getScoreMatrix(UserUser, UGTraining, numFactorsBest, lBest, mergedNetType, usersToEvaluate);
    fprintf(1,'Time elapsed: %d\n', cputime - currTime);
end

function ScoreMatrix = jointSVDAsymmetricApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    mergedNetType = 0;
    filePrefix = [Constants.LOG_PATH 'jointSVDAsym'];
    ScoreMatrix = SVDLatentFactors.getJointSVDApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, mergedNetType, filePrefix, usersToEvaluate);
end

function ScoreMatrix = jointSVDSymmetricApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    mergedNetType = 1;
    filePrefix = [Constants.LOG_PATH 'jointSVDSym'];
    [numUsers, numGroups] = size(UGTraining);
    if ~exist('usersToEvaluate')
        usersToEvaluate = 1:numUsers;
    end
    ScoreMatrix = SVDLatentFactors.getJointSVDApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, mergedNetType, filePrefix, usersToEvaluate, false, initParams);
end

function ScoreMatrix = jointClusteredSVDSymmetricApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    clusters = initParams.clusters;
    remappingKeyForClustering = initParams.remappingKeyForClustering;
    mergedNetType = 1;
    filePrefix = [Constants.LOG_PATH 'jointSVDSym'];
    [numUsers, numGroups] = size(UGTraining);
    if ~exist('usersToEvaluate')
        usersToEvaluate = 1:numUsers;
    end
    
    if(StringUtilities.isSubstring(initParams.datasetName, 'youtube'))
        % Youtube case.
        numFactors = 90;
        lBest = 1;
    else
        % Orkut case.
        numFactors = 50;
        lBest = .8;
    end
    % keyboard
    MergedNet = SVDLatentFactors.getMergedNet(UserUser, UGTraining, lBest, mergedNetType);
    [factorsForClusters, S,times relErr] = computeClusteredFactors(MergedNet, numFactors, clusters);
    relErr
    Factors = SVDLatentFactors.getFactorsFromClusters(factorsForClusters);
    
    % Set B and A.
    P = sparse(numUsers+numGroups);
    for i = 1:numUsers+numGroups
        P(i,remappingKeyForClustering(i)) = 1;
    end
    
    ScoreMatrix = P(:, usersToEvaluate)'*Factors*S*Factors'*P(:, numUsers+1:end);
end

function ScoreMatrix = jointSVDKatz(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    mergedNetType = 1;
    
    [numUsers, numGroups] = size(UGTraining);
    if ~exist('usersToEvaluate')
        usersToEvaluate = 1:numUsers;
    end
    % Set l and numFactors using best parameters learned from tKatz(C) experiments.
    if(StringUtilities.isSubstring(initParams.datasetName, 'youtube'))
        % Youtube case.
        betaExpBest = -1;
        lBest = .4;
    else
        % Orkut case.
        betaExpBest = -2;
        lBest = .2;
    end
    betaBest = 10^betaExpBest;
    numFactors = 100;
    numIterations = 3;
    
    [Factors, S] = SVDLatentFactors.mergeAndApproximate(UserUser, UGTraining, numFactors, lBest, mergedNetType);
    
    M = zeros(size(S));
    for i = 1:numIterations
        M = M + diag((betaBest*diag(S)).^i);
    end
    
    % Set B and A.
    ScoreMatrix = Factors(usersToEvaluate, :)*M*(Factors(numUsers+1:end, :)');
end

function ScoreMatrix = jointSVDKatz2(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    
    [numUsers, numGroups] = size(UGTraining);
    if ~exist('usersToEvaluate')
        usersToEvaluate = 1:numUsers;
    end
    % Set l and numFactors using best parameters learned from tKatz(C) experiments.
    if(StringUtilities.isSubstring(initParams.datasetName, 'youtube'))
        % Youtube case.
        betaExpBest = -1;
        lBest = .4;
    else
        % Orkut case.
        betaExpBest = -2;
        lBest = .2;
    end
    betaBest = 10^betaExpBest;
    numFactors = initParams.numFactors;
    numIterations = 3;
    
    [U,V,D,F] = SVDLatentFactors.combinedLowRank(UserUser,numFactors,UGTraining,numFactors);
    
    T = D*F;
    DD = F + betaBest*lBest*T+(betaBest^2*lBest*2)*D*T+betaBest^2*F*(F'*F);
    ScoreMatrix = U(usersToEvaluate, :)*DD*V';
end

function Factors = getFactorsFromClusters(factorsForClusters)
    Factors = factorsForClusters{1};
    numClusters = length(factorsForClusters);
    for(i = 2:numClusters)
        Factors = blkdiag(Factors, factorsForClusters{i});
    end
end

function ScoreMatrix = jointSVDKatzClustered(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    mergedNetType = 1;
    clusters = initParams.clusters;
    remappingKeyForClustering = initParams.remappingKeyForClustering;
    
    [numUsers, numGroups] = size(UGTraining);
    if ~exist('usersToEvaluate')
        usersToEvaluate = 1:numUsers;
    end
    % Set l and numFactors using best parameters learned from tKatz(C) experiments.
    if(StringUtilities.isSubstring(initParams.datasetName, 'youtube'))
        % Youtube case.
        betaExpBest = -1;
        lBest = .4;
    else
        % Orkut case.
        betaExpBest = -2;
        lBest = .2;
    end
    betaBest = 10^betaExpBest;
    numFactors = initParams.numFactors;
    numIterations = 3;
    
    %% Compute the clustered low rank approximation
    MergedNet = SVDLatentFactors.getMergedNet(UserUser, UGTraining, lBest, mergedNetType);
    [factorsForClusters, S,times relErr] = computeClusteredFactors(MergedNet, numFactors, clusters);
    relErr
    Factors = SVDLatentFactors.getFactorsFromClusters(factorsForClusters);

    tmpMatrix = eye(size(S, 1));
    M = zeros(size(S));
    for i = 1:numIterations
        tmpMatrix = betaBest*S*tmpMatrix;
        M = M + tmpMatrix;
    end
    
    P = sparse(numUsers+numGroups);
    for i = 1:numUsers+numGroups
        P(i,remappingKeyForClustering(i)) = 1;
    end
    % Set B and A.
    ScoreMatrix = P(:, usersToEvaluate)'*Factors*M*Factors'*P(:, numUsers+1:end);
    % whos
    % keyboard
end

function ScoreMatrix = joint2SVDSymmetricApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    mergedNetType = 2;
    filePrefix = [Constants.LOG_PATH 'joint2SVDSym'];
    ScoreMatrix = SVDLatentFactors.getJointSVDApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, mergedNetType, filePrefix, usersToEvaluate);
end

function ScoreMatrix = joint3SVDSymmetricApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    mergedNetType = 3;
    filePrefix = [Constants.LOG_PATH 'joint3SVDSym'];
    ScoreMatrix = SVDLatentFactors.getJointSVDApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, mergedNetType, filePrefix, usersToEvaluate);
end

function ScoreMatrix = joint4SVDSymmetricApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    mergedNetType = 4;
    filePrefix = [Constants.LOG_PATH 'joint4SVDSym'];
    ScoreMatrix = SVDLatentFactors.getJointSVDApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, mergedNetType, filePrefix, usersToEvaluate);
end

function ScoreMatrix = joint5SVDSymmetricApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    mergedNetType = 5;
    filePrefix = [Constants.LOG_PATH 'joint5SVDSym'];
    ScoreMatrix = SVDLatentFactors.getJointSVDApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, mergedNetType, filePrefix, usersToEvaluate);
end

function ScoreMatrix = joint6SVDSymmetricApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams, usersToEvaluate)
    import affiliationRecommendation.*;
    mergedNetType = 6;
    filePrefix = [Constants.LOG_PATH 'joint5SVDSym'];
    ScoreMatrix = SVDLatentFactors.getJointSVDApprox(UserUser, UGTraining, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, mergedNetType, filePrefix, usersToEvaluate);
end

function testClass()
    display 'Class is ok!'
end

end
end
