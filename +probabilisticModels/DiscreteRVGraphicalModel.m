classdef DiscreteRVGraphicalModel
methods(Static=true)
function Samples = getSamples(model, numSamples)
%   Sample using Mark Schmidt's UGM package. (http://www.cs.ubc.ca/~schmidtm/Software/UGM/small.html)
%   OUTPUT: Samples: numNodes * numSamples array of samples.
    import probabilisticModels.*;
    AdjMatrix = model.AdjMatrix;
    NodePotentials = model.NodePotentials;
    EdgePotentials = model.EdgePotentials;
    bIsTree = model.bIsTree;

    if(isfield(model, 'pairwiseModel'))
        % model.pairwiseModel would be the pairwise equivalent of the non-pairwise model.
        assert(strcmp(model.topology, 'spade'), 'Can only handle spades!');
        % model.pairwiseModel
        SamplesPairwise = DiscreteRVGraphicalModel.getSamples(model.pairwiseModel, numSamples);
        node1Values = mod(SamplesPairwise(1, :), model.numStates);
        node1Values(node1Values == 0) = model.numStates;
        node2Values = ceil(SamplesPairwise(1, :)/model.numStates);
        Samples = [node1Values; node2Values; SamplesPairwise(2:end, :)];
        return;
    end
    
    numStates = size(NodePotentials, 2);
    edgeStruct = UGM_makeEdgeStruct(AdjMatrix, numStates);
    if(bIsTree)
        % display 'Tree sampling: using divide and conquer'
        edgeStruct.maxIter = numSamples;
        Samples = UGM_Sample_Tree(NodePotentials, EdgePotentials, edgeStruct);
    else
%          Samples = UGM_Sample_Exact(nodePot,edgePot,edgeStruct);
        % display 'Approximate sampling: Gibbs'
        burnIn = 1000;
        edgeStruct.maxIter = numSamples;
        Samples = UGM_Sample_Gibbs(NodePotentials, EdgePotentials, edgeStruct, burnIn);
    end
end

function sampleBatches = getSampleBatches(oldSampleBatches, numSamples, numSampleSets, model)
    import probabilisticModels.*;
    if(exist('oldSampleBatches') && (size(oldSampleBatches, 1)>0))
        sampleBatches = oldSampleBatches;
        numOldSamples = size(oldSampleBatches{1}, 2);
        numSampleSets = length(oldSampleBatches);
    else
        sampleBatches = {};
        numOldSamples = 0;
    end
    for sampleBatch = 1:numSampleSets
        numNewSamples = numSamples - numOldSamples;

        if(numNewSamples > 0)
            Samples = DiscreteRVGraphicalModel.getSamples(model, numNewSamples);
            if(numOldSamples>0)
                Samples = [Samples oldSampleBatches{sampleBatch}];
            end
        else
            Samples = oldSampleBatches{sampleBatch};
            Samples = Samples(:, 1:numSamples);
        end
        sampleBatches{sampleBatch} = Samples;
    end
    fprintf('Got %d sets of %d samples! \n', numSampleSets, numSamples);
end

function Parameters = getParameters(cliqueSize, numStates, disparityLevel, parameterType, minCoupling)
    parameterDimensions = (numStates - 1)*ones(1, cliqueSize);
    switch parameterType
        case 'normal'
            ParameterTemplate = randn(parameterDimensions)*disparityLevel;
        case 'uniform'
            ParameterTemplate = rand(parameterDimensions)*disparityLevel;
    end

    Parameters = minCoupling+ParameterTemplate;
end

function [NodeParameters, EdgeParameters] = pairwiseParametersFromAdjMatrix(AdjMatrix, couplingType, numStates, disparityLevel, minCoupling)
    import probabilisticModels.*;
    if(isempty(couplingType))
        couplingType = 'normal';
    end

    numEdges = full(sum(sum(triu(AdjMatrix))));
    numNodes = size(AdjMatrix, 1);
    NodeParameters = zeros(numNodes, numStates);
    EdgeParameters = zeros(numStates, numStates, numEdges);
    for(edge = 1:numEdges)
        EdgeParameters(1:numStates-1, 1:numStates-1, edge) = DiscreteRVGraphicalModel.getParameters(2, numStates, disparityLevel, couplingType, minCoupling);
    end
end

function [model] = getDiscreteRVGraphicalModel(topology, numNodes, degmax, numStates, couplingType, disparityLevel, minCoupling)
%      Process inputs
    import graph.*;
    import probabilisticModels.*;
    AdjMatrix = Graph.getGraph(topology, numNodes, degmax);
    
    [NodeParameters, EdgeParameters] = DiscreteRVGraphicalModel.pairwiseParametersFromAdjMatrix(AdjMatrix, couplingType, numStates, disparityLevel, minCoupling);
    NodePotentials = exp(NodeParameters);
    EdgePotentials = exp(EdgeParameters);


    model.AdjMatrix = AdjMatrix;
    model.numNodes = numNodes;
    model.numStates = numStates;
    model.NodeParameters = NodeParameters;
    model.EdgeParameters = EdgeParameters;
    model.NodePotentials = NodePotentials;
    model.EdgePotentials = EdgePotentials;
    model.topology = topology;
    model.bIsTree = Graph.isTree(topology);
    model.couplingRange = [minCoupling minCoupling+disparityLevel];
    model.couplingType = couplingType;
end

function model = negateModel(modelA)
    model = modelA;
    numStates = size(model.NodePotentials, 2);
    if(isfield(model, 'NodeParameters'))
        model.NodeParameters = -modelA.NodeParameters;
    end
    model.NodePotentials = 1./modelA.NodePotentials;
    model.EdgePotentials = 1./modelA.EdgePotentials;
    model.EdgeParameters = -modelA.EdgeParameters;
    if(model.bIsTree)
        edgeStruct = UGM_makeEdgeStruct(model.AdjMatrix, numStates);
        [nodeBel, edgeBel, model.logPartitionFn] = UGM_Infer_Tree(model.NodePotentials, model.EdgePotentials, edgeStruct);
    end
end

function AdjMatrixWithEdgeIds = getAdjMatrixWithEdgeIds(AdjMatrixOriginal, numStates)
    edgeStruct = UGM_makeEdgeStruct(AdjMatrixOriginal, numStates);
    EdgeEnds = edgeStruct.edgeEnds;
    DataMatrix = [EdgeEnds (1:size(EdgeEnds, 1))'];
    numNodes = size(AdjMatrixOriginal, 1);
    AdjMatrixWithEdgeIds = MatrixTransformer.getPatternMatrix(DataMatrix, numNodes, numNodes);
    AdjMatrixWithEdgeIds = AdjMatrixWithEdgeIds + AdjMatrixWithEdgeIds';
end

function model = get3wiseDiscreteGM(topology, numNodes, numStates, parameterType, disparityLevel, minCoupling, maxParameterTriplets)
    import probabilisticModels.*;
    import graph.*;
    import probabilisticModels.*;

    [model] = DiscreteRVGraphicalModel.getDiscreteRVGraphicalModel(topology, numNodes, [], numStates, parameterType, disparityLevel, minCoupling);
    model.maxParameterTriplets = maxParameterTriplets;

    AdjMatrixWithEdgeIds = DiscreteRVGraphicalModel.getAdjMatrixWithEdgeIds(model.AdjMatrix, numStates);
    numStatesPairwise = numStates^2;

    switch topology
        case 'spade'
            TripletParameters = zeros(numStates, numStates, numStates);
            TripletParameters(1:numStates-1, 1:numStates-1, 1:numStates-1) = DiscreteRVGraphicalModel.getParameters(3, numStates, 0, 'uniform', maxParameterTriplets);
            model.tripletParameters = {TripletParameters};
            
            numNodesPairwise = numNodes -1;
            pairwiseModel = DiscreteRVGraphicalModel.getDiscreteRVGraphicalModel('chain', numNodesPairwise, [], numStatesPairwise, 'uniform', 0, -Inf);


            pairwiseModel.NodeParameters(2:end, 1:numStates) = model.NodeParameters(3:end, :);

            pairwiseModel.NodeParameters(1, :) = repmat(model.NodeParameters(1, :), 1, 3) + kron(model.NodeParameters(2, :), ones(1, 3));
            % pairwiseModel.NodeParameters(1, :)
            
            pairwiseModel.NodeParameters(2:end, numStates+1:end) = -Inf;
            pairwiseModel.NodePotentials = exp(pairwiseModel.NodeParameters);
            % pairwiseModel.NodeParameters
            % model.NodeParameters
            
            
            numPairwiseEdges = numNodesPairwise-1;
            AdjMatrixPairwiseWithEdgeIds = DiscreteRVGraphicalModel.getAdjMatrixWithEdgeIds(pairwiseModel.AdjMatrix, numStates);
            
            for(node = 2:numNodesPairwise-1)
                edge = AdjMatrixPairwiseWithEdgeIds(node, node+1);
                nodeIdOriginal = node+1;
                pairwiseModel.EdgeParameters(1:numStates, 1:numStates, edge) = model.EdgeParameters(:, :, AdjMatrixWithEdgeIds(nodeIdOriginal, nodeIdOriginal+1));
                pairwiseModel.EdgeParameters(numStatesPairwise, :, edge) = -Inf;
                pairwiseModel.EdgeParameters(:, numStatesPairwise, edge) = -Inf;
            end

            edge = AdjMatrixPairwiseWithEdgeIds(1, 2);
            pairwiseModel.EdgeParameters(numStatesPairwise, :, edge) = -Inf;
            pairwiseModel.EdgeParameters(:, numStatesPairwise, edge) = -Inf;
            
            EdgeParamsFor23 = model.EdgeParameters(:, :, AdjMatrixWithEdgeIds(2, 3));
            for node2Value=1:numStates
                EdgeParamsToAdd =  model.EdgeParameters(:, :, AdjMatrixWithEdgeIds(1, 3));
                EdgeParamsToAdd(1:numStates-1, :) =  EdgeParamsToAdd(1:numStates-1, :) + repmat(EdgeParamsFor23(node2Value, :), numStates-1, 1);
                
                pairwiseModel.EdgeParameters((node2Value-1)*numStates+1:node2Value*numStates, 1:numStates, edge) = reshape(TripletParameters(:, node2Value, :), [numStates, numStates]) + EdgeParamsToAdd;
                % display('Add pairwise parameters here.');
            end
            pairwiseModel.EdgePotentials = exp(pairwiseModel.EdgeParameters);
        otherwise
            error('Not implemented');
    end
    model.pairwiseModel = pairwiseModel;
end

function [model] = getBinaryRVGraphicalModel(topology, numNodes, degmax, couplingStrength, couplingType)
    import probabilisticModels.*;
    import graph.*;
    numStates = 2;
    model.topology = topology;
    model.couplingStrength = couplingStrength;
    model.couplingType = couplingType;
    model.maxDegree = degmax;
    
    AdjMatrix = Graph.getGraph(topology, numNodes, degmax);

    numEdges = Graph.getNumEdges(AdjMatrix);
    
    NodePotentials = ones(numNodes, numStates);
    model.EdgeParameters = zeros(numNodes);

    
    if(~isempty(findstr(topology, 'star')))
        error('Code incomplete, unverified.');
    end
    AdjMatrixWithEdgeIds = DiscreteRVGraphicalModel.getAdjMatrixWithEdgeIds(AdjMatrix, numStates);
    for(edge = 1:numEdges)
        switch couplingType
            case 'positive'
                model.EdgeParameters(AdjMatrixWithEdgeIds == edge) = couplingStrength;
            case 'negative'
                model.EdgeParameters(AdjMatrixWithEdgeIds == edge) = -couplingStrength;
            case 'mixed'
                if(randomizationUtils.Sample.binaryRandomVariable())
                    model.EdgeParameters(AdjMatrixWithEdgeIds == edge) = couplingStrength;
                else
                    model.EdgeParameters(AdjMatrixWithEdgeIds == edge) = -couplingStrength;
                end
        end
    end
    model.AdjMatrix = AdjMatrix;
    model.NodePotentials = NodePotentials;
    model = DiscreteRVGraphicalModel.fillIsingModelProperties(model);
end

function model = fillIsingModelProperties(model)
    import probabilisticModels.*;
    import graph.*;
    AdjMatrix = model.AdjMatrix;
    numEdges = Graph.getNumEdges(AdjMatrix);
    numStates = 2;
    AdjMatrixWithEdgeIds = DiscreteRVGraphicalModel.getAdjMatrixWithEdgeIds(AdjMatrix, numStates);
    model.AdjMatrixWithEdgeIds = AdjMatrixWithEdgeIds;

    EdgePotentials = zeros([numStates, numStates, numEdges]);
    for edge = 1:numEdges
        couplingStrength = model.EdgeParameters(model.AdjMatrixWithEdgeIds== edge);
        couplingStrength = couplingStrength(1);
        EdgePotentials(:, :, edge) = exp(couplingStrength*[1 -1; -1 1]);
    end
    model.EdgePotentials = EdgePotentials;

    
    model.bIsTree = Graph.isTree(AdjMatrix);
    fprintf('model.bIsTree: %d\n', model.bIsTree);
    if(model.bIsTree)
        edgeStruct = UGM_makeEdgeStruct(AdjMatrix, numStates);
        [nodeBel, edgeBel, model.logPartitionFn] = UGM_Infer_Tree(model.NodePotentials, model.EdgePotentials,edgeStruct);
    end
end

function groupIds = getGroupIndices(numNodes, numStates)
% Get variable group indices for the minimal parameters corresponding to one node of a graphical model.
    groupIds = {};
    EmptyMat = false(numStates-1, (numStates-1)*(numNodes-1)+1);
    
    Tmp = EmptyMat;
    Tmp(:,1) = true;
    groupIds{end+1} = Tmp;
    
    for(i=1:numNodes-1)
        Tmp = EmptyMat;
        Tmp(:,1+(i-1)*(numStates-1)+1:1+i*(numStates-1)) = true;
        groupIds{end+1} = Tmp;
    end
end

function [numStates, numNodes] = countStatesAndNodes(EdgeCouplings)
    numStates = size(EdgeCouplings, 1)+1;
    numNodes = (size(EdgeCouplings, 2)-1)/(numStates-1) + 1;
end

function B = getBForUnorthogonalizedFeatures(B1, Rs)
    % If X = QR, then x_{1, :} = q_{1, :}R.
    % So, q^T \beta_q = x^T R^{-1} \beta_q.
    % So, R^{-1} \beta_q = \beta_x
    % This function gets \beta_x from \beta_q.
    import probabilisticModels.*;
    [numStates, numNodes] = DiscreteRVGraphicalModel.countStatesAndNodes(B1);

    B = B1;
    for(node = 1:numNodes-1)
        nBegin = (node-1)*(numStates-1)+1+1;
        Tmp = Rs{node}\B1(:, nBegin:nBegin+(numStates-1)-1)';
        B(:, nBegin:nBegin+(numStates-2)) = Tmp';
    end
end

function B = getBForOrthogonalizedFeatures(B1, Rs)
    % This is the inverse of getBForUnorthogonalizedFeatures. See comments for that function.
    % This function gets \beta_q from \beta_x.
    import probabilisticModels.*;
    [numStates, numNodes] = DiscreteRVGraphicalModel.countStatesAndNodes(B1);

    B = B1;
    for(node = 1:numNodes-1)
        nBegin = (node-1)*(numStates-1)+1+1;
        Tmp = Rs{node}*B1(:, nBegin:nBegin+(numStates-1)-1)';
        B(:, nBegin:nBegin+(numStates-2)) = Tmp';
    end
end

function adjMatrixRow = getNeighborsFromEdgeCouplings(node, EdgeCouplings, edgeCouplingCutoff, Rs)
%   INPUT: EdgeCouplings : numStates * numStates * numNodes matrix or (numStates-1)*(numStates-1)(numNodes-1)+1 matrix.
%      Process input
    import probabilisticModels.*;
    if(~exist('edgeCouplingCutoff'))
        edgeCouplingCutoff = 0;
    end
    
    if(length(size(EdgeCouplings)) == 2)
        bMinimalParametrization = true;
        [numStates, numNodes] = DiscreteRVGraphicalModel.countStatesAndNodes(EdgeCouplings);
        groupIds = DiscreteRVGraphicalModel.getGroupIndices(numNodes, numStates);
        if(exist('Rs'))
            EdgeCouplings = DiscreteRVGraphicalModel.getBForUnorthogonalizedFeatures(EdgeCouplings, Rs);
        end
    else
        bMinimalParametrization = false;
    end
    
    if(isempty(edgeCouplingCutoff))
        edgeCouplingCutoff = 10^-4;
    end
    adjMatrixRow = zeros(1, numNodes);
    potentialNeighbors = [1:node-1, node+1:numNodes];
    for(i = 1:numNodes-1)
        j = potentialNeighbors(i);
        if(bMinimalParametrization)
            NbdEdgeCouplings = EdgeCouplings(groupIds{1+i});
        else
            NbdEdgeCouplings = EdgeCouplings(:, :, j);
        end
        NbdEdgeCouplings = NbdEdgeCouplings(:);
        if(any(abs(NbdEdgeCouplings) > edgeCouplingCutoff))
            adjMatrixRow(1, j) = 1;
        end
    end
end

function AdjMatrix = getAdjMatrixFromEdgeCouplings(EdgeCouplings, edgeCouplingCutoff, symmetrificationRule, Rs)
    import probabilisticModels.*;
    if(iscell(EdgeCouplings))
        numNodes = length(EdgeCouplings);
        bMinimalParametrization = true;
    else
        numNodes = size(EdgeCouplings, 3);
        bMinimalParametrization = false;
    end
    for(node = 1:numNodes)
        if(bMinimalParametrization)
            NbdEdgeCoupling = EdgeCouplings{node};
        else
            NbdEdgeCoupling = EdgeCouplings(:,:,:, node);
        end
        if(~exist('Rs'))
            AdjMatrix(node, :) = DiscreteRVGraphicalModel.getNeighborsFromEdgeCouplings(node, NbdEdgeCoupling, edgeCouplingCutoff, numNodes);
        else
            AdjMatrix(node, :) = DiscreteRVGraphicalModel.getNeighborsFromEdgeCouplings(node, NbdEdgeCoupling, edgeCouplingCutoff, numNodes, Rs);
        end
    end
    AdjMatrix = sparse(AdjMatrix);
    
    if(~exist('symmetrificationRule'))
        return;
    end
    
    if(strcmp(symmetrificationRule, '|'))
        AdjMatrix = AdjMatrix | AdjMatrix';
        return;
    end
    if(strcmp(symmetrificationRule, '&'))
        AdjMatrix = AdjMatrix & AdjMatrix';
    end
    
end

function [designMatrixPartBatches, RBatches] = getDesignMatrixPartBatches(sampleBatches, numStates, bGroupOrthogonalize)
    import probabilisticModels.*;

    designMatrixPartBatches = {};
    RBatches = {};
    numSampleSets = length(sampleBatches);

    for(sampleSet = 1:numSampleSets)
        Samples = sampleBatches{sampleSet};
        [designMatrixParts Rs] = DiscreteRVGraphicalModel.getDesignMatrixParts(Samples, numStates, bGroupOrthogonalize);
        designMatrixPartBatches{end+1} = designMatrixParts;
        RBatches{end+1} = Rs;
    end
end

function [designMatrixParts Rs] = getDesignMatrixParts(Samples, numStates, bGroupOrthogonalize)
    designMatrixParts = {};
    Rs = {};
    numNodes = size(Samples, 1);
    values = 1:numStates-1;
    if(~exist('bGroupOrthogonalize'))
        bGroupOrthogonalize = true;
    end
    for i=1:numNodes
        X = VectorFunctions.getIndicatorMatrixFromVector(Samples(i,:), values);
        if(bGroupOrthogonalize)
            [Q, R] = qr(double(X), 0);
            designMatrixParts{i} = Q;
            Rs{i} = R;
        else
            designMatrixParts{i} = X;
        end
    end
end

function designMatrixPartsOut = deorthogonalizeDesignMatrixParts(designMatrixParts, Rs)
    numNodes = length(designMatrixParts);
    designMatrixPartsOut = {};
    for i=1:numNodes
        designMatrixPartsOut{i} = designMatrixParts{i}*Rs{i};
    end
end

function avgNegLogPseudoLikelihood = getAvgNegLogPseudoLikelihood(Samples, EdgeCouplings, designMatrixParts)
    import probabilisticModels.*;
    [numNodes, numSamples] = size(Samples);
    if(exist('designMatrixParts'))
        bMinimalParametrization = true;
    else
        bMinimalParametrization = false;
    end
    avgNegLogPseudoLikelihood = 0;
    
    for(node=1:numNodes)
        if(bMinimalParametrization)
            [X, y] = DiscreteRVGraphicalModelLearner.getObservationsForLogisticRegression(designMatrixParts, Samples, node);

            negLogLikelihood = LogisticMultiClassL1L2Reg.getAvgNegLogLikelihoodWithDerivatives(X, y, EdgeCouplings{node});
        else
            negLogLikelihood = DiscreteRVGraphicalModelNbdLearner.getNegLogLikelihood(Samples, node, EdgeCouplings);
        end
        avgNegLogPseudoLikelihood = avgNegLogPseudoLikelihood + negLogLikelihood;
    end
    
    fprintf(' avgNegLogPseudoLikelihood: %d\n', avgNegLogPseudoLikelihood);
end

function testGetAvgNegLogPseudoLikelihood()
    import probabilisticModels.*;
    import graph.*;
    numStates = 3;
    numNodes = 3;
    degmax = [];
    disparityLevel = 3;
    couplingType = 'uniform';
    topology = 'star';
    bIsTree = 1;
    numSamples = 1000;
    [adjMatrixOriginal, NodePotentials, EdgePotentials] = DiscreteRVGraphicalModel.getDiscreteRVGraphicalModel(topology, numNodes, degmax, numStates, couplingType, disparityLevel);
    Samples = DiscreteRVGraphicalModel.getSamples(adjMatrixOriginal, NodePotentials, EdgePotentials, bIsTree, numSamples);
    EdgeCouplingsActual = zeros(numStates, numStates, numNodes, numNodes);
    EdgeCouplingsActual(:, :, 2, 1) = log(EdgePotentials(:, :, 1));
    EdgeCouplingsActual(:, :, 1, 2) = log(EdgePotentials(:, :, 1));
    EdgeCouplingsActual(:, :, 2, 3) = log(EdgePotentials(:, :, 2));
    EdgeCouplingsActual(:, :, 3, 2) = log(EdgePotentials(:, :, 2));
    
    EdgeCouplingsRandom = rand(numStates, numStates, numNodes, numNodes);
    EdgeCouplingsRandom(:, :, 1, 1) = zeros(numStates);
    EdgeCouplingsRandom(:, :, 2, 2) = zeros(numStates);
    EdgeCouplingsRandom(:, :, 3, 3) = zeros(numStates);
    avgNegLikelihoodActual = DiscreteRVGraphicalModel.getAvgNegLogPseudoLikelihood(Samples, EdgeCouplingsActual);
    avgNegLikelihoodRand = DiscreteRVGraphicalModel.getAvgNegLogPseudoLikelihood(Samples, EdgeCouplingsRandom);
    fprintf('avgNegLikelihoodActual: %d avgNegLikelihoodRand: %d, Test passed? %d \n', avgNegLikelihoodActual, avgNegLikelihoodRand, (avgNegLikelihoodActual < avgNegLikelihoodRand));
end

function testGetBinaryRVGraphicalModel()
    import probabilisticModels.*;
    import graph.*;
    numNodes = 2;
    degmax = [];
    couplingStrength = 0.999;
    couplingType = 'negative';
    topology = 'chain';
    [AdjMatrix, NodePotentials, EdgePotentials] = DiscreteRVGraphicalModel.getBinaryRVGraphicalModel(topology, numNodes, degmax, couplingStrength, couplingType)
    
    numSamples = 15;
    
    bIsTree = Graph.isTree(topology);
    Samples = DiscreteRVGraphicalModel.getSamples(AdjMatrix, NodePotentials, EdgePotentials, bIsTree, numSamples)
    
    fprintf('Expectation: In most samples, nodes connected by an edge disagree in value.');
    
    
end

function test_getNeighborsFromEdgeCouplings
    import probabilisticModels.*;
    numNodes = 3;
    numStates = 3;
    node = 1;
    EdgeCouplings = 10^(-6)*ones(numStates-1, (numNodes-1)*(numStates-1)+1);
    B = [-686.2289e-003     1.1307e-003   841.6886e-006   247.3377e-006  -110.8820e-006;
    82.8713e-003     2.4411e-003    -3.5258e-003  -163.9069e-006   -11.2823e-006];

    
    edgeCouplingCutoff = 10^(-5);
    adjMatrixRow = DiscreteRVGraphicalModel.getNeighborsFromEdgeCouplings(node, B, edgeCouplingCutoff, numNodes);
    % adjMatrixRow
    fprintf('Adj matrix row has %d edges.\n', sum(adjMatrixRow));
end

function test_getAdjMatrixFromEdgeCouplings(symmetrificationRule)
    import probabilisticModels.*;
    numNodes = 3;
    numStates = 3;
    EdgeCouplings{1} = [-444.0689e-003   -59.8775e-003    12.3910e-003   970.1192e-009    -1.5741e-006;
  -538.2851e-003    30.0785e-003    12.6506e-003    -2.0156e-006    81.2118e-009];
    EdgeCouplings{2} =   [702.4542e-003    -4.9410e-006     6.1955e-006   -11.4599e-006     6.1501e-006;
   -75.6870e-003   326.7296e-009     1.6843e-006     6.5202e-006    75.4937e-006];
    EdgeCouplings{3} = [-686.2289e-003     1.1307e-003   841.6886e-006   247.3377e-006  -110.8820e-006;
    82.8713e-003     2.4411e-003    -3.5258e-003  -163.9069e-006   -11.2823e-006];
    edgeCouplingCutoff = 10^-3;
    
    AdjMatrix = DiscreteRVGraphicalModel.getAdjMatrixFromEdgeCouplings(EdgeCouplings, edgeCouplingCutoff, symmetrificationRule);
    AdjMatrix
end

function [numStates, numNodes, Samples] = getTestSamples(numSamples)
    import probabilisticModels.*;
    numStates = 3;
    numNodes = 3;
    AdjMatrix = [0 1 0; 1 0 1; 0 1 0];
    numEdges = 2;
    model.AdjMatrix = AdjMatrix;
    model.NodePotentials = ones(numNodes);
    model.EdgePotentials = ones(numStates, numStates, 2);
    model.bIsTree = true;

    topology = 'chain';
    numNodes = 16;
    model = DiscreteRVGraphicalModel.getDiscreteRVGraphicalModel(topology, numNodes, [], numStates, 'uniform', 1, 0);

    
    Samples = DiscreteRVGraphicalModel.getSamples(model, numSamples);

end

function testGet3wiseDiscreteGM()
    import probabilisticModels.*;
    topology = 'spade';
    numNodes = 4;
    numStates = 3;
    parameterType = 'uniform';
    minCoupling = .5;
    disparityLevel = 0;
    model = DiscreteRVGraphicalModel.get3wiseDiscreteGM(topology, numNodes, numStates, parameterType, disparityLevel, minCoupling);
    Samples = DiscreteRVGraphicalModel.getSamples(model, 10000);
    unique(Samples(1, :))
    unique(Samples(2, :))
    unique(Samples(3, :))
    keyboard
end

function testClass
    display 'Class definition is ok';
end

end
end
