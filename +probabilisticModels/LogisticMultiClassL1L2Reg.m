classdef LogisticMultiClassL1L2Reg
methods(Static=true)
function [negLogLikelihood, varargout] = getNegLogLikelihoodWithDerivatives(x, y_x, B, Bx, exp_Bx)
%      Process inputs
%   Bx, exp_Bx is passed for faster execution.
    [numStatesLess1, dim] = size(B);
    % assert(length(y_x)==1, 'Bad input!');
    term1 = 0;
    normalizationFactor = 0;
    
    normalizationFactor = (1+sum(exp_Bx));
    if(y_x <=numStatesLess1)
        term1 = Bx(y_x);
    end
    negLogLikelihood = - term1 + log(normalizationFactor);

    if(nargout>1)
        GradientNegLogLikelihood = repmat(x', numStatesLess1, 1);
        HessianDiagNegLogLikelihood = GradientNegLogLikelihood.^2;
        GradientNegLogLikelihood = scale_rows(GradientNegLogLikelihood, (exp_Bx/ normalizationFactor));
        if(y_x <=numStatesLess1)
            GradientNegLogLikelihood(y_x, :) = GradientNegLogLikelihood(y_x, :) - x';
        end
        HessianDiagNegLogLikelihood = scale_rows(HessianDiagNegLogLikelihood, (exp_Bx/ normalizationFactor - (exp_Bx/ normalizationFactor).^2));
        
        varargout(1) = {GradientNegLogLikelihood};
        varargout(2) = {HessianDiagNegLogLikelihood};

        if(nargout>3)
            % Compute HessianNegLogLikelihood.
            H = -exp_Bx*exp_Bx'/(normalizationFactor^2) + diag(exp_Bx)/normalizationFactor;
            varargout(3) = {kron(H, x*x')};
        end
    end
end

function [avgNegLogLikelihood, varargout] = getAvgNegLogLikelihoodWithDerivatives(X, y, B, bUseJava)
    % Efficiency: Making the getNegLogLikelihoodWithDerivatives call inline actually decreases the speed!!
    import probabilisticModels.*;
    
%      Process inputs
    [dim, numSamples] = size(X);
    numStatesLess1 = size(B, 1);
    
    BX = B*X;
    exp_BX = exp(BX);

    if(exist('bUseJava') && bUseJava)
        % Speed decreased with the Java implementation!
        % TODO: Externalize this string.
        outputList = vvasuki.probabilisticModels.LogisticMultiClass.getAvgNegLogLikelihoodWithDerivatives(X, y, B, BX, exp_BX, nargout);
        avgNegLogLikelihood = outputList.get(0);
        if(nargout > 2)
            varargout(1) = {double(outputList.get(1))};
            varargout(2) = {double(outputList.get(2))};
        end
        if(nargout > 3),
            varargout(2) = {double(outputList.get(3))};
        end
        return;
    end

    
    avgNegLogLikelihood = 0;
    if(nargout>1)
        avgGradientNegLogLikelihood = zeros(numStatesLess1, dim);
        avgHessianDiagNegLogLikelihood = zeros(numStatesLess1, dim);
        if(nargout>3)
            avgHessianNegLogLikelihood = zeros(numStatesLess1 * dim);
        end
    end

    for i = 1:numSamples
        x = X(:, i);
        y_x = y(i);
        exp_Bx = exp_BX(:, i);
        Bx = BX(:, i);
        
        if(nargout==1)
            [NegLogLikelihood] = LogisticMultiClassL1L2Reg.getNegLogLikelihoodWithDerivatives(x, y_x, B, Bx, exp_Bx);
        else
            if(nargout <=3)
                [NegLogLikelihood, GradientNegLogLikelihood, HessianDiagNegLogLikelihood] = LogisticMultiClassL1L2Reg.getNegLogLikelihoodWithDerivatives(x, y_x, B, Bx, exp_Bx);
            else
                [NegLogLikelihood, GradientNegLogLikelihood, HessianDiagNegLogLikelihood, HessianNegLogLikelihood] = LogisticMultiClassL1L2Reg.getNegLogLikelihoodWithDerivatives(x, y_x, B, Bx, exp_Bx);
                avgHessianNegLogLikelihood = avgHessianNegLogLikelihood + HessianNegLogLikelihood;
            end
            avgGradientNegLogLikelihood = avgGradientNegLogLikelihood + GradientNegLogLikelihood;
            avgHessianDiagNegLogLikelihood = avgHessianDiagNegLogLikelihood + HessianDiagNegLogLikelihood;
        end
        avgNegLogLikelihood = avgNegLogLikelihood + NegLogLikelihood;
    end
    avgNegLogLikelihood = avgNegLogLikelihood/numSamples;
    if(nargout>1)
        avgGradientNegLogLikelihood = avgGradientNegLogLikelihood/ numSamples;
        avgHessianDiagNegLogLikelihood = avgHessianDiagNegLogLikelihood/ numSamples;
        varargout(1) = {avgGradientNegLogLikelihood};
        varargout(2) = {avgHessianDiagNegLogLikelihood};
        if(nargout>3)
            varargout(3) = {avgHessianNegLogLikelihood/numSamples};
        end
    end
%      B
%      reshape(avgGradientNegLogLikelihood, size(B))
end

function l1l2Penalty = getL1L2Penalty(B, groupsToPenalize, groupL1Penalty)
    l1l2Penalty = 0;
    numGroups = length(groupsToPenalize);
    for group = 1:numGroups
        l1l2Penalty = l1l2Penalty + norm(B(groupsToPenalize{group}), 'fro');
    end
end

function B = getBFromSamplesLagrangian(X, y, groupL1Penalty, B, groups, optimizationOptions)
    % INPUT:
    %   X: dimensions*numSamples
    %   y: 1*numSamples.
    % Solves the problem: \min nll(B) + groupL1Penalty*\norm{B}_{1;2}
    % Uses block-coordinate descent algorithm specified in "Group Lasso for Logistic Regression".
    % Orthogonalize design matrix for speed!
    % Assumes that X includes intercept.
    
    import probabilisticModels.*;
    import optimization.*;
    % Process inputs
    stoppingThreshold = optimizationOptions.stoppingThreshold;
    [dim, numSamples] = size(X);
    numStatesLess1 = size(B, 1);
    if(max(abs(B(:))) == min(abs(B(:))))
        fprintf(' The optimization logic will fail for this starting point.');
        B = rand(size(B));
    end
    % B
    
    % Declare constants
    % 1 and 2 are definitely theoretically valid. But check upper bound from proof of proposition 1.
    hessianSupportMin = 10^-5;
    groupsToPenalize = groups(2:end);
    numGroups = length(groups);
    
    
    objValPrev = Inf;
    stepSize = Inf;
    numIterations = 0;
    
    function SearchDirection = getSearchDirection(group, B, Gradient, avgHessianDiagNegLogLikelihood, stoppingThreshold, groupL1Penalty)
        groupIndices = groups{group};
        H_gg = avgHessianDiagNegLogLikelihood(groupIndices);
        hessianSupport = - max(max(H_gg(:)), hessianSupportMin);
        SearchDirection = zeros(size(B));
        if(group == 1)
            SearchDirection(groupIndices) = Gradient(groupIndices)/hessianSupport;
        else
            % Gradient'
            %% Search direction calculation.
            % Get gradient of the log likelihood fn.
            TmpMatrix = -Gradient(groupIndices) - hessianSupport*B(groupIndices);
            normTmpMatrix = norm(TmpMatrix, 'fro');
            
            if(normTmpMatrix <= groupL1Penalty)
                SearchDirection(groupIndices) = - B(groupIndices);
                % fprintf(' Case 1.\n');
            else
                SearchDirection(groupIndices) = -(-Gradient(groupIndices) - (groupL1Penalty/normTmpMatrix)*TmpMatrix)/hessianSupport;
                % fprintf(' Case 2.\n');
            end
        end
    end
    
    function stepSize = getStepSize(SearchDirection, group, B, Gradient, objFn, groupL1Penalty)
        import optimization.*;
        %% Step size calculation
        % Prepare function handles required.
        % Get gradient of the -ve log likelihood fn.
        groupIndices = groups{group};
        objFnSlice = @(stepSize)objFn(B + stepSize * SearchDirection);
        changeFnNonDifferentiablePart = @(stepSize)stepSize* groupL1Penalty*(norm(B(groupIndices) + SearchDirection(groupIndices), 'fro') - norm(B(groupIndices), 'fro'));
        stepSize = LineSearch.backtrackingSearch(objFnSlice, Gradient, SearchDirection, [], 0.75, [], changeFnNonDifferentiablePart);
    end
    
    function [B, objVal] = minimizeWrtGroup(B, group, groupL1Penalty, groupsToPenalize, stoppingThreshold, bStepLengthFromArmijoRule)
        import probabilisticModels.*;
        avgNllFn = @(BTmp) LogisticMultiClassL1L2Reg.getAvgNegLogLikelihoodWithDerivatives(X, y, BTmp, true);
        objFn = @(BTmp) avgNllFn(BTmp)+ groupL1Penalty*LogisticMultiClassL1L2Reg.getL1L2Penalty(BTmp, groupsToPenalize, groupL1Penalty);
        
        [nllVal, Gradient, avgHessianDiagNegLogLikelihood]= avgNllFn(B);
        objVal = nllVal + groupL1Penalty*LogisticMultiClassL1L2Reg.getL1L2Penalty(B, groupsToPenalize, groupL1Penalty);
        
        % fprintf(' Objective: %d [nllVal: %d], stepSizePrev: %d \n group:%d', objVal, nllVal, stepSize, group);

        SearchDirection = getSearchDirection(group, B, Gradient, avgHessianDiagNegLogLikelihood, stoppingThreshold, groupL1Penalty);
        
        if(max(abs(SearchDirection(:))) <= stoppingThreshold)
            stepSize = Inf; % For logging purposes.
            % fprintf(' group: %d, Search Direction close to 0.\n', group);
            return;
        end
        if(bStepLengthFromArmijoRule)
            stepSize = getStepSize(SearchDirection, group, B, Gradient, objFn, groupL1Penalty);
        else
            stepSize = 1;
        end
        if(stepSize == 0)
            fprintf(' 0 step size. ');
            fprintf('  group:%d Objective: %d [nllVal: %d], stepSizePrev: %d \n', group, objVal, nllVal, stepSize);
        end

        % Update B
        groupIndices = groups{group};
        B(groupIndices) = B(groupIndices) + stepSize* SearchDirection(groupIndices);
    end
    
    if(optimizationOptions.bIteratedLestSq)
        maxNumIterationsGrp = Inf;
        bStepLengthFromArmijoRule = 0;
    else
        maxNumIterationsGrp = 1;
        bStepLengthFromArmijoRule = 1;
    end

    
    while(true)
        for group = 1:numGroups
            % fprintf('getB: Considering group %d.\n', group);
            numIterations = numIterations +1;
            
            groupOptFn = @(B) minimizeWrtGroup(B, group, groupL1Penalty, groupsToPenalize, stoppingThreshold, bStepLengthFromArmijoRule);
            
            [B, objVal, numIterationsGrp] = DescentMethods.descentGeneric(groupOptFn, B, stoppingThreshold, maxNumIterationsGrp);
            
        end
        if(objValPrev - objVal <= stoppingThreshold)
            fprintf(' Objective: %d, numIterations: %d\n', objVal, numIterations);
            if(objValPrev < objVal)
                warning(' Check code! Previous Objective: %d\n', objValPrev);
                % keyboard
            end
            break;
        end
        objValPrev = objVal;
    end
end

function B = getBFromSamplesLagrangian_cvx(X, y, groupL1Penalty, B, groups, optimizationOptions)
    import probabilisticModels.*;
    [dim, numSamples] = size(X);
    groupsToPenalize = groups(2:end);
    
    cvx_begin
        variable B(size(B));
        variable t;
        minimize t + groupL1Penalty*probabilisticModels.LogisticMultiClassL1L2Reg.getL1L2Penalty(B, groupsToPenalize, groupL1Penalty);
        subject to
            LogisticMultiClassL1L2Reg.getAvgNegLogLikelihoodWithDerivatives(X, y, B) <= t;
    cvx_end
end

function testGetAvgGradientNegLogLikelihood()
    import probabilisticModels.*;
    
    node = 1;
    [numStates, numNodes, Samples] = DiscreteRVGraphicalModel.getTestSamples(1000);

    designMatrixParts = DiscreteRVGraphicalModel.getDesignMatrixParts(Samples, numStates, false);
    
    potentialNeighbors = [1:node-1, node+1:numNodes];
    X = [designMatrixParts{potentialNeighbors}]';
    X = [ones(1, size(X, 2)); X];
    y = Samples(node, :);
    
    EdgeCouplings = rand(numStates-1, (numStates-1)*(numNodes-1) + 1);
    
    
    % profile on
    timer = Timer();
    [nll, avgGradientNegLogLikelihood, avgHessianDiagNegLogLikelihood, avgHessianNegLogLikelihood] = LogisticMultiClassL1L2Reg.getAvgNegLogLikelihoodWithDerivatives(X, y, EdgeCouplings);
    timer.endTimer();
    % profile report
    % profsave(profile('info'), [gmStructureLearningExperiments.Constants.LOG_PATH 'profilingReportTest']);
    % nll
    % avgGradientNegLogLikelihood

    fprintf('Timing test.\n');
    numTrials = 5;
    for(i=1:numTrials)
        timer = Timer();
        [nll, avgGradientNegLogLikelihood, avgHessianDiagNegLogLikelihood] = LogisticMultiClassL1L2Reg.getAvgNegLogLikelihoodWithDerivatives(X, y, EdgeCouplings);
        timer.endTimer();
    end

    javaaddpath('/u/vvasuki/probabilisticModels.jar');
    javaaddpath('/u/vvasuki/colt.jar');
    for(i=1:numTrials)
        timer = Timer();
        [nll, avgGradientNegLogLikelihood, avgHessianDiagNegLogLikelihood] = LogisticMultiClassL1L2Reg.getAvgNegLogLikelihoodWithDerivatives(X, y, EdgeCouplings, true);
        timer.endTimer();
    end
    keyboard

    fprintf('Getting analytical gradient\n');
    Tmp = avgHessianDiagNegLogLikelihood';
    diag(avgHessianNegLogLikelihood)
    Tmp(:)
    fprintf('Checking consistency of the diagonal of Hessian: %d.\n', all(diag(avgHessianNegLogLikelihood) == Tmp(:)));
    
    
    fprintf('Getting numerical gradient\n');
    objFn = @(edgeCouplingsVector) LogisticMultiClassL1L2Reg.getAvgNegLogLikelihoodWithDerivatives(X, y, reshape(edgeCouplingsVector, size(EdgeCouplings)));
    [avgGradientNegLogLikelihoodNumerical errorEst] = gradest(objFn, EdgeCouplings(:));
    avgGradientNegLogLikelihoodNumerical = avgGradientNegLogLikelihoodNumerical';
    
    fprintf('Max deviation %d\n', max(abs(avgGradientNegLogLikelihood(:) - avgGradientNegLogLikelihoodNumerical(:))));
    
    avgGradientNegLogLikelihood(:)'
    avgGradientNegLogLikelihoodNumerical'
%      keyboard
    
end

function testClass
    display 'Class definition is ok';
end

end
end
