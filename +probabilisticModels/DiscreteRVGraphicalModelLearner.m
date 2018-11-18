classdef DiscreteRVGraphicalModelLearner
methods(Static=true)

function [X, y] = getObservationsForLogisticRegression(designMatrixParts, Samples, node)
    numNodes = size(Samples, 1);
    potentialNeighbors = [1:node-1, node+1:numNodes];
    X = [designMatrixParts{potentialNeighbors}]';
    X = [ones(1, size(X, 2)); X];
    y = Samples(node, :);
end

function [B sortedGroups] = thresholdParameters(B1, groupsToThreshold, threshold, node, maxUnthresholdedGroups)
    B = B1;
    % B(abs(B)< optimizationOptions.zeroingThreshold) = 0;
    grpNorms = [];
    numGroupsToThreshold = length(groupsToThreshold);
    if(~exist('maxUnthresholdedGroups'))
        maxUnthresholdedGroups = Inf;
    end
    
    for(groupNum = 1:numGroupsToThreshold)
        group = groupsToThreshold{groupNum};
        B_grp = B(group);
        grpNorms(groupNum) = norm(B_grp(:), 'fro');
        if(grpNorms(groupNum) <= threshold)
            B(group) = 0;
        end
    end
    [sortedGroupNorms sortedGroups] = sort(grpNorms, 'descend');
    for(i = maxUnthresholdedGroups+1:numGroupsToThreshold)
        group = groupsToThreshold{sortedGroups(i)};
        B(group) = 0;
    end
    
    if(exist('node'))
        sortedGroups(sortedGroups >=node) = 1 + sortedGroups(sortedGroups >=node);
    end
    
    numGroupsToPrint = min(4, numGroupsToThreshold);
    sortedGroups(1:numGroupsToPrint)
    sortedGroupNorms(1:numGroupsToPrint)
    fprintf('\n');
    
    % groupsToTruncate = find(grpNorms< max(grpNorms)/3);
    % fprintf('Truncate groups with markedly smaller norms.\n');
    % for(groupNum = groupsToTruncate)
    %     group = groupsToThreshold{groupNum};
    %     B(group) = 0;
        % fprintf('grp: %d, norm: %d\n', groupNum, grpNorms(groupNum));
    % end
end

function [B_thresholded, B] = edgeCouplingsFromSamples(Samples, designMatrixParts, Rs, groupL1Factor, EdgeCouplingsInit, optimizationOptions, node)
    import probabilisticModels.*;
    [numStates, numNodes] = DiscreteRVGraphicalModel.countStatesAndNodes(EdgeCouplingsInit);
    groups = DiscreteRVGraphicalModel.getGroupIndices(numNodes, numStates);

    fprintf('groupL1Factor: %d\n', groupL1Factor);
    B = EdgeCouplingsInit;
    if(length(Rs)>0)
        B = DiscreteRVGraphicalModel.getBForOrthogonalizedFeatures(B, Rs);
    end

    % profile on
    [X, y] = DiscreteRVGraphicalModelLearner.getObservationsForLogisticRegression(designMatrixParts, Samples, node);

    % B
    timerTmp = Timer();
    % Empirically verified that iterated least squares is faster - benefits seem to diminish with decreasing threshold.
    optimizationOptions.bIteratedLestSq = true;
    B = LogisticMultiClassL1L2Reg.getBFromSamplesLagrangian(X, y, groupL1Factor, B, groups, optimizationOptions);
    timerTmp.endTimer();

    % B
    % B = LogisticMultiClassL1L2Reg.getBFromSamplesLagrangian_cvx(X, y, groupL1Factor, B, groups, optimizationOptions);
    % B

    if(length(Rs)>0)
        B = DiscreteRVGraphicalModel.getBForUnorthogonalizedFeatures(B, Rs);
    end

    B_thresholded = DiscreteRVGraphicalModelLearner.thresholdParameters(B, groups(2:end), optimizationOptions.thresholdingFn(groupL1Factor));

    % profile report
    % profsave(profile('info'), [gmStructureLearningExperiments.Constants.LOG_PATH 'profilingReport' Timer.getTimeStamp()]);
    % keyboard
end

function EdgeCouplings = getInitEdgeCouplings(actualModel, node)
    import probabilisticModels.*;
    EdgePotentials = actualModel.EdgePotentials;
    AdjMatrixOriginal = actualModel.AdjMatrix;
    numStates = size(EdgePotentials, 2);
    numNodes = size(AdjMatrixOriginal, 1);
    bReturnRandomCouplings = false;
    if(bReturnRandomCouplings)
        EdgeCouplings = rand(numStates-1, (numStates-1)*(numNodes-1) + 1);
        return;
    end

    AdjMatrixWithEdgeIds = DiscreteRVGraphicalModel.getAdjMatrixWithEdgeIds(AdjMatrixOriginal, numStates);
    EdgeCouplings = zeros(numStates-1, (numStates-1)*(numNodes-1) + 1);
    groups = DiscreteRVGraphicalModel.getGroupIndices(numNodes, numStates);
    potentialNeighbors = [1:node-1, node+1:numNodes];
    edgeIds = AdjMatrixWithEdgeIds(node, potentialNeighbors);
    for possibleEdge = 1:numel(edgeIds)
        edgeId = edgeIds(possibleEdge);
        if(edgeId > 0)
            Values = log(EdgePotentials(1:numStates-1, 1:numStates-1, edgeId));
            EdgeCouplings(groups{1 + possibleEdge}) = Values(:);
        end
    end
end

function [results] = getTestResults(seedDataArray, experimentSettings, sampleBatches, ValidationSet, actualModel, optimizationOptions)
    import probabilisticModels.*;
    import graph.*;

    numSamplesRange = experimentSettings.numSamplesRange;
    nodesToConsider = experimentSettings.nodesToConsider;
    validationStr = experimentSettings.experimentMode;
    if(~isfield(experimentSettings, 'bOnlyThresholding'))
        experimentSettings.bOnlyThresholding = false;
    end

    % process inputs
    edgeCouplings = seedDataArray{1, 1}.edgeCouplings;
    [numStates, numNodes] = DiscreteRVGraphicalModel.countStatesAndNodes(edgeCouplings{nodesToConsider(1)});

    groups = DiscreteRVGraphicalModel.getGroupIndices(numNodes, numStates);

    function [resultsPerSample] = checkNbdRecovery(Samples, seedData)
        import probabilisticModels.*;
        import graph.*;
        edgeCouplings = seedData.edgeCouplings;
        [designMatrixParts, Rs] = DiscreteRVGraphicalModel.getDesignMatrixParts(Samples, numStates, optimizationOptions.bOrthogonalize);

        AdjMatrix = zeros(numNodes, numNodes);
        numSamples = size(Samples, 2);
        controlParam = DiscreteRVGraphicalModelLearner.controlParamForGroupL1Factor(numStates, numSamples, numNodes);
        fprintf('controlParam: %d\n', controlParam);
        if(~experimentSettings.bOnlyThresholding)
            resultsPerSample.c = seedData.c;
        end

        for node = nodesToConsider
            % Learn graph
            % display('Paused'); pause
            if(experimentSettings.bOnlyThresholding)
                % EdgeCouplingsThresholded = DiscreteRVGraphicalModelLearner.thresholdParameters(edgeCouplings{node}, groups(2:end), optimizationOptions.thresholdingFn(groupL1Factor));
                EdgeCouplingsThresholded = DiscreteRVGraphicalModelLearner.thresholdParameters(edgeCouplings{node}, groups(2:end), 0, node, sum(actualModel.AdjMatrix(node, :)));
            else
                c = resultsPerSample.c;
                groupL1Factor = c(node)*controlParam;
                [EdgeCouplingsThresholded edgeCouplings{node}] = DiscreteRVGraphicalModelLearner.edgeCouplingsFromSamples(Samples, designMatrixParts, Rs, groupL1Factor, edgeCouplings{node}, optimizationOptions, node);
            end
            AdjMatrix(node, :) = DiscreteRVGraphicalModel.getNeighborsFromEdgeCouplings(node, EdgeCouplingsThresholded);
            
            if(~experimentSettings.bOnlyThresholding)
                fprintf('node: %d, numEdges: %d\n', node, sum(AdjMatrix(node, :)));
            end
        end
        resultsPerSample.edgeCouplings = edgeCouplings;
        resultsPerSample.AdjMatrix = AdjMatrix;
    end

    function [resultsPerSampleRow] = checkNbdRecoveryRate(sampleBatches, seedDataRow)
        import probabilisticModels.*;
        if(~experimentSettings.bOnlyThresholding)
            seedData = seedDataRow{1};
            [c edgeCouplings] = DiscreteRVGraphicalModelLearner.learnConstantForGroupL1Factor({sampleBatches{1} ValidationSet}, experimentSettings, actualModel, optimizationOptions, edgeCouplings);
            seedData.c = c;
            seedData.edgeCouplings = edgeCouplings;
        end
        for(sampleSet = 1:numSampleSets)
            if(experimentSettings.bOnlyThresholding)
                seedData = seedDataRow{sampleSet};
            end
            % seedData
            resultsPerSample = checkNbdRecovery(sampleBatches{sampleSet}, seedData);
            resultsPerSampleRow{sampleSet} = resultsPerSample;
        end
    end

    numSamplesRange = sort(numSamplesRange, 'descend');
    numSampleSets = length(sampleBatches);
    numSamplesRangeSize = numel(numSamplesRange);
    for(numSamples = numSamplesRange)
        sampleBatches = DiscreteRVGraphicalModel.getSampleBatches(sampleBatches, numSamples);
        sampleSizeNum = find(numSamplesRange == numSamples);
        if(experimentSettings.bOnlyThresholding)
            seedDataRow = seedDataArray(sampleSizeNum, :);
        else
            seedDataRow = seedDataArray(1);
        end
        % keyboard
        [resultsPerSampleArray(sampleSizeNum, :)] = checkNbdRecoveryRate(sampleBatches, seedDataRow);
    end
    results.resultsPerSampleArray = resultsPerSampleArray;
end

function controlParam = controlParamForGroupL1Factor(numStates, numSamples, numNodes)
    % Used in preliminary experiments:
    % controlParam = sqrt((numStates-1)*log(numNodes)/numSamples);
    controlParam = sqrt(log(numNodes - 1)/numSamples) + (numStates-1)/(4*sqrt(numSamples));
end

function c = getCFromAlpha(alpha)
    c = 8*(2-alpha)/alpha;
end

function [c edgeCouplings] = learnConstantForGroupL1Factor(sampleBatches, experimentSettings, actualModel, optimizationOptions, edgeCouplings)
    import probabilisticModels.*;
    nodesToConsider = experimentSettings.nodesToConsider;
    AdjMatrixPartOriginal = actualModel.AdjMatrix(nodesToConsider, :);
    numNodes = size(AdjMatrixPartOriginal, 2);
    nodesToConsider = experimentSettings.nodesToConsider;
    validationStr = experimentSettings.experimentMode;
    
    %% Determine validation type.
    cRange = experimentSettings.cRange;


    c = zeros(numNodes, 1);

    function cBest = binarySearchForGroupL1Factor(sampleBatches, EdgeCouplingsInit, optimizationOptions, numEdgesOriginal, cRange, nodeToFocusOn)
        %validation to learn groupL2FactorBest
        import probabilisticModels.*;
        import graph.*;
        import optimization.*;
        fprintf('Doing binary search, numEdgesOriginal: %d\n', numEdgesOriginal);

        numSampleSetsForValidation = length(sampleBatches);
        designMatrixPartBatches = {};
        RBatches = {};
        [numStates, numNodes] = DiscreteRVGraphicalModel.countStatesAndNodes(EdgeCouplingsInit);
        groups = DiscreteRVGraphicalModel.getGroupIndices(numNodes, numStates);


        [designMatrixPartBatches, RBatches] = DiscreteRVGraphicalModel.getDesignMatrixPartBatches(sampleBatches, numStates, optimizationOptions.bOrthogonalize);

        graphFromSamplesFn = @(groupL1Factor, i, EdgeCouplings) DiscreteRVGraphicalModelLearner.edgeCouplingsFromSamples(sampleBatches{i}, designMatrixPartBatches{i}, RBatches{i}, groupL1Factor, EdgeCouplings, optimizationOptions, nodeToFocusOn);

        EdgeCouplingsEvalFn = @(EdgeCouplings, i) -Graph.getNumEdges(DiscreteRVGraphicalModel.getNeighborsFromEdgeCouplings(nodeToFocusOn, EdgeCouplings))+ numEdgesOriginal;

        function [objVal, EdgeCouplings] = tmpFn1(c, i, EdgeCouplings, graphFromSamplesFn, EdgeCouplingsEvalFn, numStates, numNodes)
            import probabilisticModels.*;
            numSamples = size(sampleBatches{i}, 2);
            controlParam = DiscreteRVGraphicalModelLearner.controlParamForGroupL1Factor(numStates, numSamples, numNodes);
            [EdgeCouplingsThresholded EdgeCouplings] = graphFromSamplesFn(c*controlParam, i, EdgeCouplings);
            objVal = EdgeCouplingsEvalFn(EdgeCouplings, i);
        end

        objFn = @(c, i, EdgeCouplings)tmpFn1(c, i, EdgeCouplings, graphFromSamplesFn, EdgeCouplingsEvalFn, numStates, numNodes);

        objFnAvg = @(c)mean(MatrixFunctions.functionalToEachColumn([1:numSampleSetsForValidation], @(i, initParams)objFn(c, i, initParams), EdgeCouplingsInit));

        val_tgt = 0;
        cBest = GlobalOptimization.binarySearch(min(cRange), max(cRange), objFnAvg, val_tgt);
    end

    function [cBest EdgeCouplings] = getMinScore_c(sampleBatches, EdgeCouplings, optimizationOptions, evalFn, cRange, node)
        import probabilisticModels.*;
        cRange = sort(cRange, 'descend');
        numSampleSets = length(sampleBatches);
        [numStates, numNodes] = DiscreteRVGraphicalModel.countStatesAndNodes(EdgeCouplings);

        [designMatrixPartBatches, RBatches] = DiscreteRVGraphicalModel.getDesignMatrixPartBatches(sampleBatches, numStates, optimizationOptions.bOrthogonalize);

        graphFromSamplesFn = @(groupL1Factor, i, EdgeCouplings) DiscreteRVGraphicalModelLearner.edgeCouplingsFromSamples(sampleBatches{i}, designMatrixPartBatches{i}, RBatches{i}, groupL1Factor, EdgeCouplings, optimizationOptions, node);


        function [objVal, EdgeCouplings] = tmpFn1(c, EdgeCouplings)
            import probabilisticModels.*;
            objValTmp = [];
            for(i = 1:numSampleSets)
                numSamples = size(sampleBatches{i}, 2);
                controlParam = DiscreteRVGraphicalModelLearner.controlParamForGroupL1Factor(numStates, numSamples, numNodes);
                [EdgeCouplingsThresholded EdgeCouplings] = graphFromSamplesFn(c*controlParam, i, EdgeCouplings);
                objValTmp(end+1) = evalFn(EdgeCouplings);
            end
            objVal = mean(objValTmp);
        end

        [scores edgeCouplingsTmp] = MatrixFunctions.functionalToEachColumn(cRange, @tmpFn1, EdgeCouplings);
        fprintf('Considering node %d.\n', node);
        scores
        cRange
        % It is possible that multiple c's have the same score.
        candidateCs = cRange(scores == min(scores));
        EdgeCouplings = edgeCouplingsTmp{cRange == candidateCs(1)};
        cBest = mean(candidateCs);
    end


    function score = edgeCouplingsEvalFn(EdgeCouplings, adjMatrixRowOriginal)
        import graph.*;
        import probabilisticModels.*;
        [extraNodesA, extraNodesB] = Graph.getDifference(adjMatrixRowOriginal, DiscreteRVGraphicalModel.getNeighborsFromEdgeCouplings(node, EdgeCouplings));
        score = extraNodesA + extraNodesB;
    end

    function c = getHardcoded_c()
        fprintf('Using hardcoded values for c.\n');
        switch(numNodes)
        case 3
            % c(node) = 0.145; % No devious thresholding necessary for this choice.
            c = 0.05*ones(numNodes, 1);
            % c(node) = 0.0;
        otherwise
            c = 0.05*ones(numNodes, 1);
            return;

            switch(actualModel.topology)
                case 'chain'
                    % c = 0.3;
                    c(node) = 0.15;
                case 'grid'
                    c = zeros(numNodes, 1);
                    c(:) = 0.876;
                    c([1,4,13, 16]) = .414;
                    c([2, 3, 5, 8, 9, 12, 14, 15]) = 0.293;
                    return;
            end
        end
    end
    
    if(StringUtilities.isSubstring(validationStr, 'Likelihood'))
        fprintf(' Doing max likelihood.\n');
        [numStates, numNodes] = DiscreteRVGraphicalModel.countStatesAndNodes(edgeCouplings{nodesToConsider(1)});
        numSamplesRange = cellfun(@(S)size(S, 2), sampleBatches);
        maxSetIndex = find(numSamplesRange == max(numSamplesRange));
        maxSetIndex = maxSetIndex(1);
        ValidationSamples = sampleBatches{maxSetIndex};
        sampleBatches = sampleBatches([1:maxSetIndex-1 maxSetIndex+1:length(sampleBatches)]);
    end
    for(node = nodesToConsider)
        EdgeCouplings = edgeCouplings{node};
        adjMatrixRowOriginal = AdjMatrixPartOriginal(nodesToConsider == node, :);
        if(~exist('validationStr') || length(validationStr) == 0)
            c(node) = DiscreteRVGraphicalModelLearner.getCFromAlpha(actualModel.alpha);
        elseif(StringUtilities.isSubstring(validationStr, 'OldExp'))
            c = getHardcoded_c();
        elseif(StringUtilities.isSubstring(validationStr, 'binary'))
            fprintf(' Doing binary search.\n');
            cRange = [0, max(cRange)];
            c(node) = binarySearchForGroupL1Factor(sampleBatches, EdgeCouplings, optimizationOptions, full(sum(adjMatrixRowOriginal)), cRange, node);
        elseif(StringUtilities.isSubstring(validationStr, 'Likelihood'))
            fprintf(' Doing max likelihood.\n');
            
            [designMatrixParts Rs] = DiscreteRVGraphicalModel.getDesignMatrixParts(ValidationSamples, numStates, false);
            [X, y] = DiscreteRVGraphicalModelLearner.getObservationsForLogisticRegression(designMatrixParts, ValidationSamples, node);
            
            avgNegLogLikelihoodFn = @(B)LogisticMultiClassL1L2Reg.getAvgNegLogLikelihoodWithDerivatives(X, y, B);
            [c(node), edgeCouplings{node}] = getMinScore_c(sampleBatches, EdgeCouplings, optimizationOptions, avgNegLogLikelihoodFn, cRange, node);
        else
            fprintf(' Doing min-edge-disagreement.\n');
            [c(node), edgeCouplings{node}] = getMinScore_c(sampleBatches, EdgeCouplings, optimizationOptions, @(B)edgeCouplingsEvalFn(B, adjMatrixRowOriginal), cRange, node);
            % error('Not implemented');
        end
     end
    fprintf('Best c: ');
    c'
end

function [model] = checkModelLearnability(edgeCouplings, Samples, model, nodesToConsider)
    display 'Checking learnability';
    import probabilisticModels.*;
    AdjMatrix = model.AdjMatrix;
    model.bLearnability = true;
    [numStates, numNodes] = DiscreteRVGraphicalModel.countStatesAndNodes(edgeCouplings{nodesToConsider(1)});
    [designMatrixParts Rs] = DiscreteRVGraphicalModel.getDesignMatrixParts(Samples, numStates, false);
    
    groups = DiscreteRVGraphicalModel.getGroupIndices(numNodes, numStates);
    EventMatrix = [designMatrixParts{1:numNodes}]';
    CooccuranceProbability = statistics.Estimation.estimateCooccuranceProbabilities(EventMatrix);
    
    function [ew_min, incoherence, ew_max_XX] = checkNbdLearnability(node)
        import probabilisticModels.*;
        EdgeCouplings = edgeCouplings{node};
        adjMatrixRow = AdjMatrix(node, :);
        [X, y] = DiscreteRVGraphicalModelLearner.getObservationsForLogisticRegression(designMatrixParts, Samples, node);
        
        [nll, avgGradientNegLogLikelihood, avgHessianDiagNegLogLikelihood, ExpectedHessianNll] = LogisticMultiClassL1L2Reg.getAvgNegLogLikelihoodWithDerivatives(X, y, EdgeCouplings);

        % keyboard

        dim = size(X, 1);
        potentialNeighbors = [1:node-1, node+1:numNodes];
        nbdSet = find(adjMatrixRow(potentialNeighbors));
        nonNbdSet = find(~adjMatrixRow(potentialNeighbors));
        function indexTemplate = getIndexTemplate(nodeSet)
            indexTemplate = zeros(dim, 1);
            for neighbor = nodeSet
                tmp = groups{1 + neighbor};
                tmp = tmp(1, :);
                indexTemplate = indexTemplate | tmp';
            end
        end
        
        nbdSetIndex = kron(ones(numStates-1, 1), getIndexTemplate(nbdSet));
        nbdSetIndex = logical(nbdSetIndex);

        % The below yields error sometimes.
        % ew_min = eigs(ExpectedHessianNll(nbdSetIndex, nbdSetIndex), 1, 'SM');
        [V, D] = eig(ExpectedHessianNll(nbdSetIndex, nbdSetIndex));
        ew_min = min(diag(D));
        
        nonNbdSetIndex = kron(ones(numStates-1, 1), getIndexTemplate(nonNbdSet));
        nonNbdSetIndex = logical(nonNbdSetIndex);
        Tmp = ExpectedHessianNll(nonNbdSetIndex, nbdSetIndex)/ExpectedHessianNll(nbdSetIndex, nbdSetIndex);
        incoherence = norm(Tmp, 'inf');
        if(isnan(incoherence))
            warning('incoherence is nan! Probably the edgeCouplings had nan values, and edgePotentials had Inf values.');
            Tmp
            keyboard
        end

        tmp = zeros(numNodes, 1);
        tmp(potentialNeighbors) = 1;
        nbdIndices = kron(tmp, ones(numStates-1, 1));
        % keyboard
        nbdIndices = logical(nbdIndices);
        CooccuranceProbabilityOtherNodes = CooccuranceProbability(nbdIndices, nbdIndices);
        % CooccuranceProbabilityOtherNodes
        ew_max_XX = eigs(CooccuranceProbabilityOtherNodes, 1, 'LM');
    end
    model.ew_minBound = +Inf;
    model.incoherenceBound = -Inf;
    model.ew_max_XXBound = -Inf;

    for(node = nodesToConsider)
        [ew_min, incoherence, ew_max_XX] = checkNbdLearnability(node);
        
        model.ew_minBound = min([model.ew_minBound, ew_min]);
        model.incoherenceBound = max([incoherence, model.incoherenceBound]);
        model.ew_max_XXBound = max([model.ew_max_XXBound, ew_max_XX]);
        fprintf('ew_min: %d, incoherence: %d, ew_max_XX: %d\n', ew_min, incoherence, ew_max_XX);
    end
    fprintf('ew_minBound: %d, incoherenceBound: %d, ew_max_XXBound: %d\n', model.ew_minBound, model.incoherenceBound, model.ew_max_XXBound);

    %% Note that alpha's relation with the population Fisher Information Matrix is different from its relation with the sample FIsher information matrix.
    model.alpha = (1-model.incoherenceBound);
    if(model.alpha <= 0)
        warning('Invalid alpha: %d!\n', model.alpha);
        model.bLearnability = false;
        % error('Invalid alpha!');
    end

    numSamplesMin = 50;
    lambdaMax = DiscreteRVGraphicalModelLearner.getCFromAlpha(model.alpha)*DiscreteRVGraphicalModelLearner.controlParamForGroupL1Factor(numStates, numSamplesMin, numNodes);

    numSamplesMax = 2^13;
    lambdaMin = DiscreteRVGraphicalModelLearner.getCFromAlpha(model.alpha)*DiscreteRVGraphicalModelLearner.controlParamForGroupL1Factor(numStates, numSamplesMax, numNodes);

%      if(10*lambdaMax/model.ew_minBound > (max(model.couplingRange)))
%          warning('10*lambdaMax/model.ew_minBound is too big: %d!\n', 10*lambdaMax/model.ew_minBound);
%          10*lambdaMax/model.ew_minBound
%          max(model.couplingRange)
%          model.bLearnability = false;
%          % error('Edge recovery not possible!');
%      end
%      if(10*lambdaMin/model.ew_minBound > (max(model.couplingRange)))
%          warning('10*lambdaMin/model.ew_minBound is too big: %d!\n', 10*lambdaMin/model.ew_minBound);
%          10*lambdaMin/model.ew_minBound
%          max(model.couplingRange)
%          model.bLearnability = false;
%          % error('Edge recovery not possible!');
%      end
end

function testNbdFromSamples(numNodesOrFileName, topology, nodesToConsider, experimentMode)
    % INPUT:
    %  experimentMode: Choices below are described using regular expressions.
    %%    'parallelPrepare-VALIDATION' - Prepare a parallel experiment. VALIDATION is one of the validation choices described below.
    %           Requires: numNodes, topology, nodesToConsider
    %           Provides: actualModel, experimentName, sampleBatches, ValidationSet, seedDataArray stored in a mat file. Detailed commands for executing parallel jobs stored in a batch file.
    %%    'parallelRun-VALIDATION' - Run a parallel experiment, parallel. VALIDATION is one of the validation choices described below.
    %           Requires: nodesToConsider as an argument. actualModel, experimentName, sampleBatches, ValidationSet, seedDataArray stored in a mat file.
    %           Provides: The results strucutre stored in a mat file.
    %%    'parallelCombine-VALIDATION' - combine the results from several parallel runs.
    %           Requires: actualModel, experimentName, sampleBatches, ValidationSet, seedDataArray stored in a mat file prepared by parallelPrepare - that file name should be passed in the arguments. It will then seek various result files prepared using parallelRun.
    %           Provides: The results strucutre stored in a mat file.
    %
    %
    %     Validation choices:
    %      ''
    %      '*Binary*' - Use binary search.
    %      '*Likelihood*' - Use max-likelihood.
    %      '.+' - (Any non-empty string different from the above.) Use minimum edge disagreement.
    
    import probabilisticModels.*;
    import graph.*;


    if(~exist('experimentMode'))
        experimentMode = '';
    end
    
    %% Get probability of success plot
    experimentSettings.numSamplesRange = 10^3*[2:1:12];
    % experimentSettings.numSamplesRange = 500*[1:5];
    experimentSettings.cRange = [.05:0.05:0.15];
    experimentSettings.numSamplesRange = sort(experimentSettings.numSamplesRange, 'descend');
    experimentSettings.numSampleSets = 15;
    experimentSettings.experimentMode = experimentMode;
    experimentSettings.nodesToConsider = nodesToConsider;
    experimentSettings.symmetrificationRule = '';
    experimentSettings.maxParameterTriplets = 10^-5;
    experimentSettings.numParallelPrograms = 10;

    % Settings for parallelPrepare

    % Get samples
    %% Prepare the sample batches for validation
    function [sampleBatches, ValidationSet] = getSampleBatches(actualModel)
        sampleBatchGeneratorFn = @(numSamples, numSampleSets, sampleBatches)DiscreteRVGraphicalModel.getSampleBatches(sampleBatches, numSamples, numSampleSets, actualModel);

        timerTmp = Timer();
        sampleBatches = {};
        sampleBatches = sampleBatchGeneratorFn(max(experimentSettings.numSamplesRange), experimentSettings.numSampleSets, sampleBatches);
        timerTmp.endTimer();

        ValidationSet = [];
        if(length(experimentSettings.experimentMode) > 0)
            ValidationSet = sampleBatchGeneratorFn(max(experimentSettings.numSamplesRange)*2, 1, {});
            ValidationSet = ValidationSet{1};
        end
    end

    function [strB] = appendNodeFocusDetails(strB, nodesToConsider)
        strB = [strB '-' strrep(num2str(nodesToConsider([1, numel(nodesToConsider)])), '  ', 'n') 'nd'];
    end

    function [actualModel, edgeCouplings] = testModelAndGetEdgeCouplings(actualModel)
        import probabilisticModels.*;
        edgeCouplings = {};
        for node = experimentSettings.nodesToConsider
            edgeCouplings{node} = DiscreteRVGraphicalModelLearner.getInitEdgeCouplings(actualModel, node);
            % edgeCouplings{node}
        end
        % log(actualModel.EdgePotentials)

        
        if(~StringUtilities.isSubstring(experimentSettings.experimentMode, 'parallelRun') && ~StringUtilities.isSubstring(experimentSettings.experimentMode, 'parallelCombine'))
            TmpSamples = DiscreteRVGraphicalModel.getSamples(actualModel, 2^12);
            actualModel.bLearnability = true;
            % TODO: TEMP
            % [actualModel] = DiscreteRVGraphicalModelLearner.checkModelLearnability(edgeCouplings, TmpSamples, actualModel, experimentSettings.nodesToConsider);
        end
    end
    
    function [actualModel, experimentName, sampleBatches, ValidationSet, seedDataArray] = getRandomGraphicalModel(numNodes, topology, nodesToConsider)
        import probabilisticModels.*;
        %% Get graphical model.
        numStates = 3;
        % numStates = numNodes;
        degmax = [];
        disparityLevel = 0;
        assert(~StringUtilities.isSubstring(experimentMode, 'parallelCombine'), 'Must pass file name, not number of nodes if you use parallelCombine!');

        % minCoupling = 1;
        couplingType = 'uniform';
        if(~exist('topology', 'var') || isempty(topology))
            % topology = 'chain';
            topology = 'star';
        end
        %% Set up penalty type.
        if(~exist('groupL1penaltyType') || isempty(groupL1penaltyType))
            groupL1penaltyType = 'L1L2';
            % groupL1penaltyType = 'L1L2Lagrangian';
            % groupL1penaltyType = 'L1L_inf';
        end
        for minCoupling = 0.5
            if(strcmp(topology, 'spade'))
                actualModel = DiscreteRVGraphicalModel.get3wiseDiscreteGM(topology, numNodes, numStates, couplingType, disparityLevel, minCoupling, experimentSettings.maxParameterTriplets);
            else
                [actualModel] = DiscreteRVGraphicalModel.getDiscreteRVGraphicalModel(topology, numNodes, degmax, numStates, couplingType, disparityLevel, minCoupling);
            end
            [actualModel, edgeCouplings] = testModelAndGetEdgeCouplings(actualModel);
            if(actualModel.bLearnability)
                break;
            end
        end
        if(~actualModel.bLearnability)
            error('Could not find a learnable model!');
        end
        actualModel.numStates = numStates;
        actualModel.numNodes = numNodes;
        seedDataArray{1, 1}.edgeCouplings = edgeCouplings;

        experimentName = [actualModel.topology num2str(numNodes) 'n-' num2str(actualModel.couplingRange(1)) ':' num2str(actualModel.couplingRange(2)) actualModel.couplingType groupL1penaltyType];
        experimentName = appendNodeFocusDetails(experimentName, nodesToConsider);
        

        % log(EdgePotentials)
        [sampleBatches, ValidationSet] = getSampleBatches(actualModel);
    end

    function [actualModel, experimentName, sampleBatches, ValidationSet, seedDataArray, fileNameSansExtension] = loadGraphicalModelFromFile(fileNameSansExtension)
        import probabilisticModels.*;
        
        if(fileNameSansExtension(1) == '/')
            fileNameSansExtension = [fileNameSansExtension];
        else
            if(StringUtilities.isSubstring(experimentSettings.experimentMode, 'parallel'))
                filePath = gmStructureLearningExperiments.Constants.TMP_PATH;
            else
                filePath = gmStructureLearningExperiments.Constants.LOG_PATH;
            end
            fileNameSansExtension = [filePath fileNameSansExtension];
        end
        s = load([fileNameSansExtension '.mat']);
        actualModel=s.actualModel;
        if(isfield(s, 'experimentName'))
            experimentName=s.experimentName;
        else
            experimentName=s.experimentSettings.experimentName;
        end
        sampleBatches=s.sampleBatches;
        ValidationSet=s.ValidationSet;
        % seedDataArray = results.resultsPerSampleArray;
        [actualModel, edgeCouplings] = testModelAndGetEdgeCouplings(actualModel);
        seedDataArray{1, 1}.edgeCouplings = edgeCouplings;

        if(StringUtilities.isSubstring(experimentSettings.experimentMode, 'parallel'))
            experimentSettings.numSamplesRange=s.experimentSettings.numSamplesRange;
        end
        if(~exist('sampleBatches'))
            [sampleBatches, ValidationSet] = getSampleBatches(actualModel);
        end
    end


    if(length(numNodesOrFileName)>1)
        fileNameSansExtension = numNodesOrFileName;
        [actualModel, experimentName, sampleBatches, ValidationSet, seedDataArray, fileNameSansExtension] = loadGraphicalModelFromFile(fileNameSansExtension);
    else
        [actualModel, experimentName, sampleBatches, ValidationSet, seedDataArray] = getRandomGraphicalModel(numNodesOrFileName, topology, experimentSettings.nodesToConsider);
    end
    experimentSettings.experimentName = experimentName;
    % keyboard

    if(StringUtilities.isSubstring(experimentSettings.experimentMode, 'parallelPrepare'))
        dataFile = [gmStructureLearningExperiments.Constants.TMP_PATH experimentSettings.experimentName '-parallelPrepare' Timer.getTimeStamp()];
        %% Append experiment details (adjascency matrix, couplings) to mat file.
        save([dataFile '.mat'],'actualModel', 'sampleBatches', 'ValidationSet', 'experimentSettings');

        experimentModeForCommand = strrep(experimentSettings.experimentMode, 'parallelPrepare', 'parallelRun');
        commandFileName = [dataFile '-parallelCommands.sh'];
        commandFileName
        commandFile = fopen(commandFileName, 'w');
        machines = system.Parallelization.getTargetMachines();
        nodesToConsiderParts = VectorFunctions.getRoughlyEqualParts(nodesToConsider, experimentSettings.numParallelPrograms);
        % keyboard
        for(part = 1:length(nodesToConsiderParts))
            nodesToConsiderPart = nodesToConsiderParts{part};
            machineName = machines{part};
            fileNameStub = strrep(dataFile, 'parallelPrepare', 'parallelResults');
            fileNameStub = appendNodeFocusDetails(fileNameStub, nodesToConsiderPart);
            matlabCommand = sprintf('probabilisticModels.DiscreteRVGraphicalModelLearner.testNbdFromSamples(''%s'',''%s'', %s, ''%s'')', dataFile, topology, ['[' num2str(nodesToConsiderPart) ']'], experimentModeForCommand);
            jobStr = system.Parallelization.prepareMatlabJob(matlabCommand, fileNameStub, machineName);
            % fprintf('%s\n', jobStr);
            fprintf(commandFile, '%s\n', jobStr);
        end
        
        fclose(commandFile);
        system(['chmod a+x ' commandFileName]);
        
        % error('Run parallel condor experiments');
        return;
    end

    optimizationOptions.stoppingThreshold = 10^-7;
    % Empirically verified that orthogonalization does not increase speed - it essentially has no effect.
    optimizationOptions.bOrthogonalize = false;
    function threshold = thresholdingRule(groupL1Factor)
        % threshold = groupL1Factor/actualModel.ew_minBound;
        threshold = 2*groupL1Factor;
        % threshold = (3/4)*max(actualModel.couplingRange);
    end
    optimizationOptions.thresholdingFn = @thresholdingRule;
    

    if(StringUtilities.isSubstring(experimentSettings.experimentMode, 'parallel'))
        resultsFile = strrep(fileNameSansExtension, 'parallelPrepare', 'parallelResults');
    end

    function results = combineParallelResults()
        import probabilisticModels.*;
        function [resultsList] = getStoredResults()
            files = IO.listFiles([resultsFile '*.mat']);
            numFiles = length(files);
            fprintf('Found %d result files!\n', numFiles);

            resultsList = {};
            for(fileNum = 1:numFiles)
                tmp = load(files{fileNum});
                if(isfield(tmp, 'nodesToConsider'))
                    tmp.results.nodesToConsider = tmp.nodesToConsider;
                else
                    tmp.results.nodesToConsider = tmp.experimentSettings.nodesToConsider;
                    tmp.results.experimentSettings = tmp.experimentSettings;
                end
                resultsList{end+1} = tmp.results;
            end
        end
        function results = combine2Results(resultA, resultB)
            function resultsPerSample = combineResultsPerSample(resultsPerSampleA, resultsPerSampleB, nodesConsideredB)
                resultsPerSample = resultsPerSampleA;
                if(isfield(resultsPerSample, 'AdjMatrix'))
                    resultsPerSample.AdjMatrix =  resultsPerSample.AdjMatrix | resultsPerSampleB.AdjMatrix;
                end
                for(node = nodesConsideredB)
                    resultsPerSample.c(node) = resultsPerSampleB.c(node);
                    resultsPerSample.edgeCouplings{node} = resultsPerSampleB.edgeCouplings{node};
                end
            end
            results = resultA;
            % fprintf('Combining results\n');
            results.nodesToConsider = [results.nodesToConsider resultB.nodesToConsider];
            % keyboard
            for(i = 1:numel(results.resultsPerSampleArray))
                results.resultsPerSampleArray{i} = combineResultsPerSample(results.resultsPerSampleArray{i}, resultB.resultsPerSampleArray{i}, resultB.nodesToConsider);
            end
        end
        function results = combineResults(resultsList)
            import probabilisticModels.*;
            numResults = length(resultsList);
            results = resultsList{1};
            for(resultNum = 2:numResults)
                % fprintf('resultNum: %d \n', resultNum);
                results = combine2Results(results, resultsList{resultNum});
            end
        end
        [resultsList] = getStoredResults();
        
        results = combineResults(resultsList);
        results.nodesToConsider = sort(results.nodesToConsider);
        results.experimentSettings = resultsList{1}.experimentSettings;
        experimentSettings.nodesToConsider = nodesToConsider;
        % experimentSettings.numSamples = results.experimentSettings.numSamples;
        fprintf('Combined results! Nodes present: \n');
        results.nodesToConsider
    end

    function inspectResultsPerSample(nodesToConsider, resultPerSample)
        import probabilisticModels.*;
        edgeCouplings = resultPerSample.edgeCouplings;
        [numStates, numNodes] = DiscreteRVGraphicalModel.countStatesAndNodes(edgeCouplings{nodesToConsider(1)});
        groups = DiscreteRVGraphicalModel.getGroupIndices(numNodes, numStates);
        for(node = nodesToConsider)
            fprintf('Considering neighborhood of node %d \n', node);
            DiscreteRVGraphicalModelLearner.thresholdParameters(edgeCouplings{node}, groups(2:end), -Inf, node);
        end
    end

    function SuccessMatrix = getSuccessMatrix(nodesToConsider, resultsPerSampleArray)
        function bSuccess = checkSuccess(nodesToConsider, AdjMatrix)
            import graph.*;
            AdjMatrixPartOriginal = actualModel.AdjMatrix(nodesToConsider, :);
            AdjMatrixPart = AdjMatrix(nodesToConsider, :);
            [extraEdges, extraEdgesInOriginal] = Graph.getDifference(AdjMatrixPart, AdjMatrixPartOriginal);
            numEdges = Graph.getNumEdges(AdjMatrixPart);
            fprintf('numEdges: %d, extraEdges: %d, extraEdgesInOriginal: %d\n', numEdges, extraEdges, extraEdgesInOriginal);
            bSuccess = (extraEdges + extraEdgesInOriginal == 0);
        end
        SuccessMatrix = false(size(resultsPerSampleArray));
        strcmp(experimentSettings.symmetrificationRule, '|')
        strcmp(experimentSettings.symmetrificationRule, '&')
        for(i = 1:numel(resultsPerSampleArray))
            AdjMatrix = resultsPerSampleArray{i}.AdjMatrix;
            if(strcmp(experimentSettings.symmetrificationRule, '|'))
                AdjMatrix = AdjMatrix | AdjMatrix';
            end
            if(strcmp(experimentSettings.symmetrificationRule, '&'))
                AdjMatrix = AdjMatrix & AdjMatrix';
            end
            SuccessMatrix(i) = checkSuccess(nodesToConsider, AdjMatrix);
        end
    end
    
    function [fullFileNameSansExtension, probabilitiesOfSuccess] = plotProbabilitySuccess(SuccessMatrix, nodesConsidered)
        probabilitiesOfSuccess = sum(SuccessMatrix, 2)/ experimentSettings.numSampleSets;
        if(~exist('nodesConsidered'))
            nodesConsidered = experimentSettings.nodesToConsider;
        end
        if(numel(nodesConsidered) == max(nodesConsidered))
            title = '';
        else
            title = num2str(nodesConsidered);
        end
        title = [title experimentSettings.symmetrificationRule];
        % title = [title num2str(optimizationOptions.threshold)];
        [figureHandle, fullFileNameSansExtension] = IO.plotAndSave(experimentSettings.numSamplesRange(1:numel(probabilitiesOfSuccess)), probabilitiesOfSuccess, 'numSamples', 'Prob. of success',  gmStructureLearningExperiments.Constants.LOG_PATH, experimentSettings.experimentName, title);

        %% Append experiment details (adjascency matrix, couplings) to mat file.
        save([fullFileNameSansExtension '.mat'],'actualModel', 'experimentName', 'experimentSettings', 'results', 'optimizationOptions', '-append');
        if(exist('topology'))
            save([fullFileNameSansExtension '.mat'], 'topology', '-append');
        end
    end

    if(StringUtilities.isSubstring(experimentSettings.experimentMode, 'parallelCombine'))
        results = combineParallelResults();
        fnTmp = @(i, j)inspectResultsPerSample(results.nodesToConsider, results.resultsPerSampleArray{i, j});
        
        experimentSettings.bOnlyThresholding = true;
        experimentSettings.nodesToConsider = results.nodesToConsider;
        seedDataArray = results.resultsPerSampleArray;
        [results] = DiscreteRVGraphicalModelLearner.getTestResults(seedDataArray, experimentSettings, sampleBatches, ValidationSet, actualModel, optimizationOptions);
        
        
    else
        timer = Timer();
        [results] = DiscreteRVGraphicalModelLearner.getTestResults(seedDataArray, experimentSettings, sampleBatches, ValidationSet, actualModel, optimizationOptions);
        timer.endTimer();
        
        if(StringUtilities.isSubstring(experimentSettings.experimentMode, 'parallelRun'))
            resultsFile = appendNodeFocusDetails(resultsFile, nodesToConsider);
            fprintf('parallelRun : Saving results!\n');
            save([resultsFile '.mat'], 'results', 'experimentSettings', 'optimizationOptions');
            return;
        end
        warning('Recomputing results using aggressive thresholding.');
        experimentSettings.bOnlyThresholding = true;
        [results] = DiscreteRVGraphicalModelLearner.getTestResults(results.resultsPerSampleArray, experimentSettings, sampleBatches, ValidationSet, actualModel, optimizationOptions);
    end
    
    % keyboard
    % experimentSettings.nodesToConsider = 8;
    function getProbabilityOfSuccessCurve(nodesToConsiderTmp)
        SuccessMatrix = getSuccessMatrix(nodesToConsiderTmp, results.resultsPerSampleArray);
        [fullFileNameSansExtension, probabilitiesOfSuccess] = plotProbabilitySuccess(SuccessMatrix);
    end
    getProbabilityOfSuccessCurve(experimentSettings.nodesToConsider)
    fprintf('All done: ready for inspection!\n');
    keyboard

end

function generateFigures
    filePath = gmStructureLearningExperiments.Constants.LOG_PATH;
    filePrefix = filePath;

    xLabel = 'numSamples';
    yLabel = 'Prob. of success';
    
    % Scaling x axis.
    xLabelNew = 'Control parameter';
    dLogpScaler = @(x, maxDegree, numNodes, k)x/(k*maxDegree*log(numNodes));
    constantScaler = @(x, y) x/y;

    fileName = [filePath 'spade32nNode3'];
    maxDegree = 3;
    numNodes = 32;
    k = 10;
    IO.replaceAxis(fileName, 1, xLabelNew, @(x)dLogpScaler(x, maxDegree, numNodes, k));
    IO.removeLegendsAndTitle(fileName);

    fileName = [filePath 'chain32n'];
    maxDegree = 2;
    numNodes = 32;
    k = 10;
    IO.replaceAxis(fileName, 1, xLabelNew, @(x)dLogpScaler(x, maxDegree, numNodes, k));
    IO.removeLegendsAndTitle(fileName);

    fileName = [filePath 'chain64nNode16'];
    maxDegree = 2;
    numNodes = 64;
    k = 10;
    IO.replaceAxis(fileName, 1, xLabelNew, @(x)dLogpScaler(x, maxDegree, numNodes, k));
    IO.removeLegendsAndTitle(fileName);

%      fileName = [filePath 'chain16n4nd1'];
%      maxDegree = 2;
%      numNodes = 16;
%      k = 10;
%      IO.replaceAxis(fileName, 1, xLabelNew, @(x)dLogpScaler(x, maxDegree, numNodes, k));
%      IO.removeLegendsAndTitle(fileName);

    fileName = [filePath 'grid25nNode7'];
    maxDegree = 4;
    numNodes = 25;
    k = 10;
    IO.replaceAxis(fileName, 1, xLabelNew, @(x)dLogpScaler(x, maxDegree, numNodes, k), yLabel);
    IO.removeLegendsAndTitle(fileName);

    figureName = 'nbdRecoveryGrid';
    files = {[filePath 'grid16nNode6.mat'] [filePath 'grid25nNode7.mat']};
    legendNames = {'p=16, node 6' 'p=25, node 7'};
    [figureHandle, fullFileNameSansExtension] = IO.getFiguresAndCombine(files, filePrefix, figureName, xLabelNew, yLabel, legendNames);


    figureName = 'nbdRecovery-chain';
    files = {[filePath 'chain64nNode16.mat'] [filePath 'chain32nNode8.mat'] [filePath 'chain16n4nd1.mat']};
    legendNames = {'p=64, node 16' 'p=32, node 8' 'p=16, node 4' };
    [figureHandle, fullFileNameSansExtension] = IO.getFiguresAndCombine(files, filePrefix, figureName, [], [], legendNames);
    
    figureName = 'fullStructureRecovery';
    files = {[filePath 'chain16n.mat'] [filePath 'chain32n.mat']};
    legendNames = {'p=16' 'p=32'};
    [figureHandle, fullFileNameSansExtension] = IO.getFiguresAndCombine(files, filePrefix, figureName, xLabelNew, yLabel, legendNames);

    % fileName = [filePath 'nbdRecovery-chain'];
    % IO.replaceAxis(fileName, 1, xLabelNew, @(x)constantScaler(x, 10));

end

function testClass
    display 'Class definition is ok';
end

end
end
