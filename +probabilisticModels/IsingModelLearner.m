classdef IsingModelLearner
methods(Static=true)

function [y, X, potentialNeighbors] = getLogisticRegressionInput(node, Samples)
    numNodes = size(Samples, 1);
    potentialNeighbors = setdiff(1:numNodes,[node]);
    potentialNeighbors = potentialNeighbors(:);
    y = Samples(node, :)';
    X = Samples(potentialNeighbors, :);
end

function avgNegLogPseudoLikelihood = getAvgNegLogPseudoLikelihood(Samples, model)
    import probabilisticModels.*;
    [numNodes, numSamples] = size(Samples);
    avgNegLogLikelihoods = [];
    for node = 1:numNodes
        [y, X, potentialNeighbors] = IsingModelLearner.getLogisticRegressionInput(node, Samples);
        beta = [model.EdgeParameters(node, node) model.EdgeParameters(node, potentialNeighbors)]';
        beta = beta(2:end);
        avgNegLogLikelihoods(end+1) = LogLinearModels.getAvgNegLogLikelihoodWithDerivatives(y, X, beta);
        
    end
    avgNegLogPseudoLikelihood = mean(avgNegLogLikelihoods);
end

function [model] = isingModelFromSamples(Samples, lambda, thresholdingFn)
    import probabilisticModels.*;
    import graph.*;
    xRange = unique(Samples);
    assert(min(xRange ) == -1 && numel(xRange) == 2, 'Improper Samples.');
    [numNodes, numSamples] = size(Samples);
    EdgeCouplings = zeros(numNodes);
    

    function learnNodeNbd(node, lambdaNd)
        import probabilisticModels.*;
        [y, X, potentialNeighbors] = IsingModelLearner.getLogisticRegressionInput(node, Samples);
        
        beta = LogLinearModels.logisticRegressionl1Reg(y, X, lambdaNd);
        % Do thresholding as optimization problems is not solved to complete accuracy.
        beta(2:end) = thresholdingFn(beta(2:end), node);
        % As beta corresponds to 2*EdgeCouplings, we do the following:
        beta = beta/2;
        EdgeCouplings(node, potentialNeighbors) = beta(2:end);
        EdgeCouplings(node, node) = beta(1);
        
    end
    
    for node = 1:numNodes
        learnNodeNbd(node, lambda);
    end

    % Symmetrification:
    EdgeCouplings = 0.5*(EdgeCouplings + EdgeCouplings');

    AdjMatrix = (EdgeCouplings ~= 0);
    AdjMatrix = AdjMatrix - diag(diag(AdjMatrix));
    numEdges = Graph.getNumEdges(AdjMatrix);
    numStates = 2;
    model.NodePotentials = ones(numNodes, numStates);
    model.AdjMatrix = AdjMatrix;
    model.EdgeParameters = EdgeCouplings;
    model = DiscreteRVGraphicalModel.fillIsingModelProperties(model);
end

function model = isingModelFromSamplesAutopickLambda(Samples, cRange, numSplits, actualModel)
    import probabilisticModels.*;

    function treenessPenalty = getTreenessPenalty(AdjMatrix)
        import graph.*;
        treenessPenalty = 0;
        if(~Graph.isTree(AdjMatrix))
            treenessPenalty = Inf;
        end
    end

    function v = pickTopFew(v)
        threshold = max(abs(v))/3;
        smallEntries = abs(v) < threshold;
        v(smallEntries) = 0;
    end

    function v = pickNodesUsingDegreeKnowledge(v, node)
        AdjMatrix = actualModel.AdjMatrix;
        numNodes = size(AdjMatrix, 1);
        numNbd = sum(AdjMatrix(node, :));
        % fprintf('%d has %d neighbors.\n', node, numNbd);
        v = MatrixTransformer.retainTopEntries(v, numNbd);
    end

    learner = @(c, Samples)IsingModelLearner.isingModelFromSamples(Samples, LogLinearModels.lambdaFromC(c, Samples), @pickNodesUsingDegreeKnowledge);

    negLikelihoods = [];

    function likelihood = evaluateC(c)
        import statistics.*;
        import probabilisticModels.*;
        learnerGivenC = @(Samples)learner(c, Samples);

        tester = @(model, TestSet)IsingModelLearner.getAvgNegLogPseudoLikelihood(TestSet, model);

        likelihood = Estimation.getGeneralizationAbilityCrossValidation(learnerGivenC, tester, numSplits, Samples);
    end
    
    for(c = cRange)
        negLikelihoods(end+1) = evaluateC(c);
    end
    negLikelihoods
    cBest = negLikelihoods(find(negLikelihoods == min(negLikelihoods), 1));
    model = learner(cBest, Samples);
end

function [model] = checkModelLearnability(model, Samples)
    display 'Checking learnability';
    import probabilisticModels.*;
    EdgeParameters = model.EdgeParameters;
    AdjMatrix = model.AdjMatrix;
    numNodes = size(EdgeParameters, 1);

    beta = [model.logPartitionFn; EdgeParameters(logical(triu(ones(size(EdgeParameters)), 1)))];
    alphaOfNodes = [];
    for node=1:numNodes
        y = Samples(node, :)';
        potentialNeighbors = [1:node-1 node+1:numNodes]';
        X = Samples(potentialNeighbors, :);

        edgeParametersNode = EdgeParameters(node, potentialNeighbors);
        beta = 2*edgeParametersNode;
        % Find avgHessian
        [avgNegLogLikelihood, avgHessian] = LogLinearModels.getAvgNegLogLikelihoodWithDerivatives(y, X, beta);
        avgHessian = 4*avgHessian;
        % keyboard

        actualNeighbors = logical(AdjMatrix(node, :)');
        actualNeighbors(node) = [];
        actualNonNeighbors = ~actualNeighbors;
        alphaOfNodes(end+1) = 1- norm(avgHessian(actualNonNeighbors, actualNeighbors)/avgHessian(actualNeighbors, actualNeighbors), inf);
        % Find incoherence
    end
    model.alpha = min(alphaOfNodes);
    model.alphaOfNodes = alphaOfNodes;
    model.c = 16*(2 - model.alpha)/model.alpha;

    alphaOfNodes
    fprintf('c is %d\n', model.c);

end

function [SpanTree] = bestTreeModelGraph(Sample)
% Implements the Chow Liu algorithm.
    import statistics.*;
    import graph.*;
    MutualInfo = Estimation.getMutualInformationMatrix(Sample);
    SpanTree = Graph.getMaxSpanningTree(MutualInfo);
end

function [model] = treeModelFromSamples(Samples)
    import probabilisticModels.*;
    import graph.*;
    [AdjMat] = IsingModelLearner.bestTreeModelGraph(Samples);
    model.AdjMatrix = logical(AdjMat);

    [numNodes, numSamples] = size(Samples);
    EdgeCouplings = zeros(numNodes);


    function learnNodeNbd(node)
        import probabilisticModels.*;
        neighborsIndex = model.AdjMatrix(:, node);
        numNeighbors = sum(neighborsIndex);
        if(numNeighbors == 0)
            return;
        end
        
        y = Samples(node, :);
        X = Samples(neighborsIndex, :);

        beta = LogLinearModels.logisticRegressionl1Reg(y, X, 0);
        % As beta corresponds to 2*EdgeCouplings, we do the following:
        beta = beta/2;
        
        EdgeCouplings(node, neighborsIndex ) = beta(2:end);
        EdgeCouplings(node, node) = beta(1);
    end
    for node = 1:numNodes
        learnNodeNbd(node);
    end

    % Symmetrification:
    EdgeCouplings = 0.5*(EdgeCouplings + EdgeCouplings')

    numStates = 2;
    model.NodePotentials = ones(numNodes, numStates);
    model.EdgeParameters = EdgeCouplings;
    model = DiscreteRVGraphicalModel.fillIsingModelProperties(model);
end

function testClass
    display 'Class definition is ok';
end

end
end
