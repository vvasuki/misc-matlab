classdef DiscreteRVGraphicalModelNbdLearner
methods(Static=true)
function [negLogLikelihood, varargout] = getNegLogLikelihoodWithDerivatives(sample, node, EdgeCouplings)
%      Process inputs
    [numStates, b, numNodes] = size(EdgeCouplings);
    
    GradientNegLogLikelihood = zeros(numStates, numStates, numNodes);
    potentialNeighbors = [1:node-1 node+1:numNodes]';

    term1 = 0;
    sumExp = 0;
    normalizationFactor = 0;
    onesTmp = ones(numNodes - 1, 1);
    for(k = 1:numStates)
        indicesTmp = subv2ind(size(EdgeCouplings), [sample(potentialNeighbors), k*onesTmp, potentialNeighbors]);
        
        sumCouplings = sum(EdgeCouplings(indicesTmp));
        if(k == sample(node))
            term1 = sumCouplings;
        end

        conditionalProbabilityFactor = exp(sumCouplings );
        
        normalizationFactor = normalizationFactor + conditionalProbabilityFactor;

        if(nargout>1)
            GradientNegLogLikelihood(indicesTmp) = conditionalProbabilityFactor;
        end
    end
    negLogLikelihood = - term1 + log(normalizationFactor);
    
    if(nargout>1)
        GradientNegLogLikelihood = GradientNegLogLikelihood/ normalizationFactor;
        
        % Adjust gradients corersponding to sample(node), sample(potentialNeighbors), potentialNeighbors.
        indicesTmp = subv2ind(size(EdgeCouplings), [sample(potentialNeighbors), sample(node)*onesTmp, potentialNeighbors]);
        GradientNegLogLikelihood(indicesTmp) = -1 + GradientNegLogLikelihood(indicesTmp);

        HessianDiagNegLogLikelihood = -GradientNegLogLikelihood.^2 + GradientNegLogLikelihood;
        varargout(1) = {GradientNegLogLikelihood};
        varargout(2) = {HessianDiagNegLogLikelihood};
    end
end

function [avgNegLogLikelihood, varargout] = getAvgNegLogLikelihoodWithDerivatives(Samples, node, EdgeCouplings)
    import probabilisticModels.*;
    
%      Process inputs
    [numNodes, numSamples] = size(Samples);
    numStates = size(EdgeCouplings, 1);
    
    
    avgGradientNegLogLikelihood = zeros(numStates, numStates, numNodes);
    avgHessianDiagNegLogLikelihood = zeros(numStates, numStates, numNodes);
    avgNegLogLikelihood = 0;
    for i = 1:numSamples
        sample = Samples(:, i);
        
        if(nargout==1)
            [NegLogLikelihood] = DiscreteRVGraphicalModelNbdLearner.getNegLogLikelihoodWithDerivatives(sample, node, EdgeCouplings);
        else
            [NegLogLikelihood, GradientNegLogLikelihood, HessianDiagNegLogLikelihood] = DiscreteRVGraphicalModelNbdLearner.getNegLogLikelihoodWithDerivatives(sample, node, EdgeCouplings);
            
            avgGradientNegLogLikelihood = avgGradientNegLogLikelihood + GradientNegLogLikelihood;
            avgHessianDiagNegLogLikelihood = avgHessianDiagNegLogLikelihood + HessianDiagNegLogLikelihood;
        end
        avgNegLogLikelihood = avgNegLogLikelihood + NegLogLikelihood;
    end
    avgNegLogLikelihood = avgNegLogLikelihood/numSamples;
    if(nargout>1)
        avgGradientNegLogLikelihood = avgGradientNegLogLikelihood(:)/ numSamples;
        avgHessianDiagNegLogLikelihood = avgHessianDiagNegLogLikelihood(:)/ numSamples;
        varargout(1) = {avgGradientNegLogLikelihood};
        varargout(2) = {avgHessianDiagNegLogLikelihood};
    end
%      EdgeCouplings
%      reshape(avgGradientNegLogLikelihood, size(EdgeCouplings))
end


function l1l2Penalty = getL1L2Penalty(NbdEdgeCouplings, groupL1Penalty)
    l1l2Penalty = 0;
    numNodes = size(NbdEdgeCouplings, 3);
    for node = 1:numNodes
        l1l2Penalty = l1l2Penalty + norm(NbdEdgeCouplings(:, :, node), 'fro');
    end
end

function avgGradientNegLogLikelihood = getAvgGradientNegLogLikelihood(Samples, node, EdgeCouplings)
    import probabilisticModels.*;
    [avgNegLogLikelihood, avgGradientNegLogLikelihood, avgHessianDiagNegLogLikelihood] = DiscreteRVGraphicalModelNbdLearner.getAvgNegLogLikelihoodWithDerivatives(Samples, node, EdgeCouplings);
end

function NbdEdgeCoupling = getNbdEdgeCouplingFromSamples(Samples, node, groupL1Bound, groupL1penaltyType, NbdEdgeCoupling)
    import probabilisticModels.*;
    % Process inputs
    [numNodes, numSamples] = size(Samples);
    numStates = size(NbdEdgeCoupling, 1);
    fprintf('Considering node %d.\n', node);
    if(isempty(groupL1penaltyType))
    %    groupL1penaltyType = 'L1L_inf';
        groupL1penaltyType = 'L1L2';
    end
    
    avgNegLogLikelihoodFn = @(edgeCouplingsVector)DiscreteRVGraphicalModelNbdLearner.getAvgNegLogLikelihoodWithDerivatives(Samples, node, reshape(edgeCouplingsVector, numStates, numStates, numNodes));
    avgGradientNegLogLikelihoodFn = @(edgeCouplingsVector)DiscreteRVGraphicalModelNbdLearner.getAvgGradientNegLogLikelihood(Samples, node, reshape(edgeCouplingsVector, numStates, numStates, numNodes));
    funObj = @(edgeCouplingsVector)FunctionUtilities.combine2functions(avgNegLogLikelihoodFn, avgGradientNegLogLikelihoodFn, edgeCouplingsVector);
    
    groupIdentifiers = zeros(numStates, numStates, numNodes);
    for(n = 1:numNodes)
        groupIdentifiers(:, :, n) = n;
    end
    groupIdentifiers = groupIdentifiers(:);
    
    if(StringUtilities.isSubstring(groupL1penaltyType, 'inf'))
        fprintf('L1L_inf projection\n');
        projectorFn = @(edgeCouplingsVector)groupLinfProj(edgeCouplingsVector,groupL1Bound, groupIdentifiers);
    else
        fprintf('L1L2 projection\n');
        projectorFn = @(edgeCouplingsVector)groupL2Proj(edgeCouplingsVector,groupL1Bound, groupIdentifiers);
    end
    options = struct('verbose', 1);
    NbdEdgeCoupling(:) = minConF_PQN(funObj, NbdEdgeCoupling(:), projectorFn, options);
    
end

function NbdEdgeCoupling = getNbdEdgeCouplingFromSamplesLagrangian(Samples, node, groupL1Penalty, NbdEdgeCoupling, optimizationOptions)
    import probabilisticModels.*;
    import optimization.*;
    stoppingThreshold = optimizationOptions.stoppingThreshold;
    % Solves the problem: \min nll(NbdEdgeCoupling) + groupL1Penalty*\norm{NbdEdgeCoupling}_{1;2}
    % Uses block-coordinate descent algorithm specified in "Group Lasso for Logistic Regression".
    % Process inputs
    [numNodes, numSamples] = size(Samples);
    numStates = size(NbdEdgeCoupling, 1);
    
    fprintf('getNbdEdgeCoupling: Considering node %d.\n', node);
    if(max(abs(NbdEdgeCoupling(:))) == min(abs(NbdEdgeCoupling(:)))
        fprintf(' The optimization logic will fail for this starting point.');
        NbdEdgeCoupling = rand(size(NbdEdgeCoupling));
    end
    assert(), ' The optimization logic will fail for this starting point.');
    % NbdEdgeCoupling
    
    % Declare constants
    % 1 and 2 are definitely theoretically valid. But check upper bound from proof of proposition 1.
    hessianSupportMin = 10^-5;
    
    avgNllFn = @(NbdEdgeCouplingsTmp) DiscreteRVGraphicalModelNbdLearner.getAvgNegLogLikelihoodWithDerivatives(Samples, node, NbdEdgeCouplingsTmp);
    objFn = @(NbdEdgeCouplingsTmp) avgNllFn(NbdEdgeCouplingsTmp)+ groupL1Penalty*DiscreteRVGraphicalModelNbdLearner.getL1L2Penalty(NbdEdgeCouplingsTmp, groupL1Penalty);
    
    objValPrev = Inf;
    while(true)
        for neighbor = 1:numNodes
            if(neighbor == node)
                continue;
            end
            [objVal, Gradient, avgHessianDiagNegLogLikelihood]= avgNllFn(NbdEdgeCoupling);
            objVal = objVal + groupL1Penalty*DiscreteRVGraphicalModelNbdLearner.getL1L2Penalty(NbdEdgeCoupling, groupL1Penalty);
            % fprintf('getNbdEdgeCoupling: Considering neighbor %d.\n', neighbor);
            
            hessianSupport = - max(max(avgHessianDiagNegLogLikelihood), hessianSupportMin);
            % Gradient'

            Gradient = reshape(Gradient, numStates, numStates, numNodes);
            % NbdEdgeCoupling

            %% Search direction calculation.
            NbdEdgeCouplingSearchDirection = zeros(numStates, numStates, numNodes);
            % Get gradient of the log likelihood fn.
            Gradient = -Gradient;
            TmpMatrix = Gradient(:, :, neighbor) - hessianSupport*NbdEdgeCoupling(:, :, neighbor);
            normTmpMatrix = norm(TmpMatrix, 'fro');
            
            if(normTmpMatrix <= groupL1Penalty)
                NbdEdgeCouplingSearchDirection(:, :, neighbor) = - NbdEdgeCoupling(:, :, neighbor);
                % fprintf(' Case 1.\n');
            else
                NbdEdgeCouplingSearchDirection(:, :, neighbor) = -(Gradient(:,:, neighbor) - (groupL1Penalty/normTmpMatrix)*TmpMatrix)/hessianSupport;
                % fprintf(' Case 2.\n');
            end
            % fprintf('getNbdEdgeCoupling: Found search direction \n');
            if(max(abs(NbdEdgeCouplingSearchDirection(:)'*Gradient(:))) <= stoppingThreshold || max(abs(NbdEdgeCouplingSearchDirection(:))) <= stoppingThreshold)
                fprintf(' Directional derivative too small.\n');
                continue;
            end
            
            
            %% Step size calculation
            % Prepare function handles required.
            % Get gradient of the -ve log likelihood fn.
            Gradient = -Gradient;
            objFnSlice = @(stepSize)objFn(NbdEdgeCoupling + stepSize * NbdEdgeCouplingSearchDirection);
            changeFnNonDifferentiablePart = @(stepSize)stepSize* groupL1Penalty*(norm(NbdEdgeCoupling(:, :, neighbor) + NbdEdgeCouplingSearchDirection(:, :, neighbor), 'fro') - norm(NbdEdgeCoupling(:, :, neighbor), 'fro'));
            stepSize = LineSearch.backtrackingSearch(objFnSlice, Gradient, NbdEdgeCouplingSearchDirection, [], 0.75, [], changeFnNonDifferentiablePart);
            
            % Update NbdEdgeCoupling
            NbdEdgeCoupling = NbdEdgeCoupling + stepSize* NbdEdgeCouplingSearchDirection;
            % fprintf(' Objective: %d, stepSize: %d \n', objVal, stepSize);
        end
        if(objValPrev - objVal <= stoppingThreshold)
            fprintf(' Objective: %d\n', objVal);
            if(objValPrev < objVal)
                error(' Check code! Previous Objective: %d\n', objValPrev);
            end
            break;
        end
        objValPrev = objVal;
    end

end

function NbdEdgeCoupling = getNbdEdgeCouplingFromSamplesLagrangian_cvx(Samples, node, groupL1Penalty, NbdEdgeCoupling, optimizationOptions)
    import probabilisticModels.*;
    stoppingThreshold = optimizationOptions.stoppingThreshold;
    [numNodes, numSamples] = size(Samples);
    
    cvx_begin
        variable NbdEdgeCoupling(size(NbdEdgeCoupling));
        variable negLogLikelihoods(numSamples);
        minimize mean(negLogLikelihoods) + groupL1Penalty*probabilisticModels.DiscreteRVGraphicalModelNbdLearner.getL1L2Penalty(NbdEdgeCoupling, groupL1Penalty);
        subject to
        for(i=1:numSamples)
            DiscreteRVGraphicalModelNbdLearner.getNegLogLikelihoodWithDerivatives(Samples(:,i), node, NbdEdgeCoupling) <= negLogLikelihoods(i);
        end
    cvx_end
end


function [EdgeCouplings] = edgeCouplingsFromSamples(Samples, numStates, groupL1Factor, groupL1penaltyType, EdgeCouplings, optimizationOptions)
    import probabilisticModels.*;
    numNodes = size(Samples, 1);
    timesElapsed = [];
    for(node = 1:numNodes)
        NbdEdgeCoupling = EdgeCouplings(:, :, :, node);
        timer = Timer();
    profile on
        if(StringUtilities.isSubstring(groupL1penaltyType, 'Lagrangian'))
            % NbdEdgeCoupling
            NbdEdgeCoupling = DiscreteRVGraphicalModelNbdLearner.getNbdEdgeCouplingFromSamplesLagrangian(Samples, node, groupL1Factor, NbdEdgeCoupling, optimizationOptions);
            % NbdEdgeCoupling
            % NbdEdgeCoupling = DiscreteRVGraphicalModelNbdLearner.getNbdEdgeCouplingFromSamplesLagrangian_cvx(Samples, node, groupL1Factor, NbdEdgeCoupling, optimizationOptions);
        else
            NbdEdgeCoupling = DiscreteRVGraphicalModelNbdLearner.getNbdEdgeCouplingFromSamples(Samples, node, groupL1Factor, groupL1penaltyType, NbdEdgeCoupling);
        end
        
        NbdEdgeCoupling(abs(NbdEdgeCoupling)< optimizationOptions.zeroingThreshold) = 0;
        nllVal = DiscreteRVGraphicalModelNbdLearner.getAvgNegLogLikelihoodWithDerivatives(Samples, node, NbdEdgeCouplings);
        objVal = nllVal + groupL1Factor*DiscreteRVGraphicalModelNbdLearner.getL1L2Penalty(NbdEdgeCouplings, groupL1Factor);
        fprintf(' After thresholding. Objective: %d [nllVal: %d]\n', objVal, nllVal);
        
    profile report
        timer = timer.endTimer();
        timesElapsed(end+1) = timer.elapsedTime;
        EdgeCouplings(:, :, :, node) = NbdEdgeCoupling;
        % NbdEdgeCoupling
        % keyboard
    end
    fprintf(' Average time taken: %d.\n', mean(timesElapsed));
    
    % EdgeCouplings
end

function groupL1FactorBest = binarySearchForGroupL1Factor(sampleBatches, numStates, groupL1penaltyType, EdgeCouplingsInit, optimizationOptions, numEdgesOriginal, groupL1FactorRange, symmetrificationRule)
    import probabilisticModels.*;
    import graph.*;
    import optimization.*;
    %validation to learn groupL2FactorBest
    
    graphFromSamplesFn = @(groupL1Factor, TrainingSamples) DiscreteRVGraphicalModelLearner.edgeCouplingsFromSamples(TrainingSamples, numStates, groupL1Factor, groupL1penaltyType, EdgeCouplingsInit, optimizationOptions);
    
    objFn = @(groupL1Factor, i) -sum(sum(DiscreteRVGraphicalModel.getAdjMatrixFromEdgeCouplings(graphFromSamplesFn(groupL1Factor, sampleBatches{i}), 0, symmetrificationRule)))/2 + numEdgesOriginal;
    
    objFnAvg = @(groupL1Factor)mean(MatrixFunctions.functionalToEachColumn([1:5], @(i)objFn(groupL1Factor, i)));
    
    val_tgt = 0;
    groupL1FactorBest = GlobalOptimization.binarySearch(min(groupL1FactorRange), max(groupL1FactorRange), objFnAvg, val_tgt);
end

function testGetNegLogLikelihood()
    import probabilisticModels.*;
    sample = [1; 2; 3];
    numStates = 3;
    numNodes = 3;
    node = 1;
    EdgeCouplings = zeros(numStates, numStates, numNodes);
    negLogLikelihood = DiscreteRVGraphicalModelNbdLearner.getAvgNegLogLikelihoodWithDerivatives(sample, node, EdgeCouplings);
    fprintf('Conditional probability of sample(node): %d; expected: %d \n', exp(-negLogLikelihood), 1/numStates);
end

function testGetAvgGradientNegLogLikelihood()
    import probabilisticModels.*;
    
    node = 1;
    [numStates, numNodes, Samples] = DiscreteRVGraphicalModel.getTestSamples;
    
    EdgeCouplings = rand(numStates, numStates, numNodes);
    
    
    fprintf('Getting analytical gradient\n');
    [nll, avgGradientNegLogLikelihood] = DiscreteRVGraphicalModelNbdLearner.getAvgNegLogLikelihoodWithDerivatives(Samples, node, EdgeCouplings);
    
    fprintf('Getting numerical gradient\n');
    objFn = @(edgeCouplingsVector) DiscreteRVGraphicalModelNbdLearner.getAvgNegLogLikelihoodWithDerivatives(Samples, node, reshape(edgeCouplingsVector, numStates, numStates, numNodes));
    [avgGradientNegLogLikelihoodNumerical errorEst] = gradest(objFn, EdgeCouplings(:));
    avgGradientNegLogLikelihoodNumerical = avgGradientNegLogLikelihoodNumerical';
    
    fprintf('Max deviation %d\n', max(abs(avgGradientNegLogLikelihood - avgGradientNegLogLikelihoodNumerical)));
    avgGradientNegLogLikelihood(:)'
    avgGradientNegLogLikelihoodNumerical'
%      keyboard
    
end

function testGetGradientNegLogLikelihood()
    import probabilisticModels.*;
    
    node = 1;
    [numStates, numNodes, Samples] = DiscreteRVGraphicalModel.getTestSamples;
    
    
    EdgeCouplings = rand(numStates, numStates, numNodes);
    
    
    fprintf('Getting analytical gradient\n');
    [nll, gradientNegLogLikelihood] = DiscreteRVGraphicalModelNbdLearner.getNegLogLikelihoodWithDerivatives(Samples, node, EdgeCouplings, false);
    
    fprintf('Getting numerical gradient\n');
    objFn = @(edgeCouplingsVector) DiscreteRVGraphicalModelNbdLearner.getNegLogLikelihoodWithDerivatives(Samples, node, reshape(edgeCouplingsVector, numStates, numStates, numNodes));
    [gradientNegLogLikelihoodNumerical errorEst] = gradest(objFn, EdgeCouplings(:));
    gradientNegLogLikelihoodNumerical = gradientNegLogLikelihoodNumerical';
    fprintf('Max deviation %d\n', max(abs(gradientNegLogLikelihood(:) - gradientNegLogLikelihoodNumerical)));
%      gradientNegLogLikelihood(:)'
%      gradientNegLogLikelihoodNumerical'
    
end

function testClass
    display 'Class definition is ok';
end

end
end
