classdef LogLinearModels
methods(Static=true)
function [beta] = logisticRegressionl1Reg(y, Samples, lambda)
% Model: Pr(y=1) = exp(\beta_0 + \sum_i beta_i x_i)/(1+exp(\beta_0 + \sum_i beta_i x_i))
% Choices for y: +1 or -1.
% Problem solved: min_b avgNll(b|Samples, y) + lambda*||b||_1, where avgNll is the average negative log likelihood function.
%INPUT: RESPONSE Y, PREDICTOR Samples, PENALTY LAMBDA
%OUTPUT: COEFFICIENTS BETA.
% Uses code by Boyd. Wrapper by pradeepr.
    Samples = Samples';
    y = y(:);
    % whos
    n = size(Samples,1);
    p = size(Samples,2);
    % toleranceDuality = 10^(-10);
    % fprintf('Input details: %d, %d, %d\n', n, p, lambda);
    Xfile = sprintf('/tmp/Samples-%d-%d-%.2f',p,n,lambda);
    yfile = sprintf('/tmp/y-%d-%d-%.2f',p,n,lambda);
    modelfile = sprintf('/tmp/model-%d-%d-%.2f',p,n,lambda);
    
    mmwrite(Xfile,Samples);
    mmwrite(yfile,y);
    system([system.SystemUtils.LIBRARIES_PATH sprintf( '/optimization/l1_logreg-0.8.2-i686-pc-linux-gnu/l1_logreg_train -q -s %s %s %f %s', Xfile, yfile, lambda, modelfile)]);
    beta = full(mmread(modelfile));
    % beta = beta(2:end);
    fprintf('lambda %d, non-zeros: %d \n', lambda, sum(beta(2:end) ~= 0));
    % beta'
end

function [avgNegLogLikelihood, varargout] = getAvgNegLogLikelihoodWithDerivatives(y, X, b)
    % Model: Pr(y=1) = exp(\beta_0 + \sum_i beta_i x_i)/(1+exp(\beta_0 + \sum_i beta_i x_i))
    % Choices for y: +1 or -1.
    b = b(:);
    bX = b'*X;
    avgNegLogLikelihood = mean(log(1+exp(bX))-bX);

    if(nargout>1)
        numSamples = size(X, 2);
        avgHessianNegLogLikelihood = 0;
        for i=1:numSamples
            x = X(:, i);
            avgHessianNegLogLikelihood = avgHessianNegLogLikelihood + (exp(y(i)*dot(x, b))/(1+exp(y(i)*dot(x, b)))^2)*x*x';
        end
        avgHessianNegLogLikelihood = avgHessianNegLogLikelihood/numSamples;
        varargout(1) = {avgHessianNegLogLikelihood};
    end
end

function lambdaRange = lambdaFromC(cRange, Samples)
    lambdaRange = sqrt(log(size(Samples, 1))/size(Samples, 2))*cRange;
end

function [beta, cBest] = logisticRegressionl1RegWithValidation(y, Samples, cRange, numSplits)
    import probabilisticModels.*;
    import statistics.*;
    import optimization.*;
    if(~exist('numSplits', 'var'))
        numSplits = 10;
    end
    
    learnerParametrized = @(X1, y1, c)Classifier.classifierFromParameters(@LogLinearModels.classifyLogistic, LogLinearModels.logisticRegressionl1Reg(y1, X1, LogLinearModels.lambdaFromC(c, X1)));

    badnessFn = @(c) ClassificationEvaluation.getMisclassificationRateFromCrossValidation(Samples, y, @(X1, y1)learnerParametrized(X1, y1, c), numSplits);

    % TEMPORARY.
    % badnessFn = @(lambda) -ClassificationEvaluation.getClassifierMisclassificationRate(learnerParametrized(Samples, y, lambda), Samples, y);
    
    [objMin, cBest] = GlobalOptimization.exhaustiveSearchScalarFn(cRange, badnessFn);
    
    beta = LogLinearModels.logisticRegressionl1Reg(y, Samples, LogLinearModels.lambdaFromC(cBest, Samples));
    fprintf('logisticRegressionl1RegWithValidation: Found beta!\n');
    beta'
end

function label = classifyLogistic(x, beta, bIgnoreBiasTerm)
% Model: Pr(y=1) = exp(\beta_0 + \sum_i beta_i x_i)/(1+exp(\beta_0 + \sum_i beta_i x_i))
    % Labels are +1 or -1!
    % beta'
    if(exist('bIgnoreBiasTerm') && bIgnoreBiasTerm)
        beta = beta(2:end);
    end
    if (numel(x) == numel(beta))
        label = (dot(x,beta) >= 0);
    elseif (numel(x) == numel(beta) - 1)
        label = ((dot(x, beta(2:end)) + beta(1)) >= 0);
    else
        error('Invalid input! dim(x) is %d, but dim(beta) is %d', numel(x), numel(beta));
    end
    if(label == 0)
        label = -1;
    end
end

function labels = classifyLogisticEnMasse(Samples, beta)
% Model: Pr(y=1) = exp(\beta_0 + \sum_i beta_i x_i)/(1+exp(\beta_0 + \sum_i beta_i x_i))
    import probabilisticModels.*;
    labels = MatrixFunctions.functionalToEachColumn(Samples, @(x)LogLinearModels.classifyLogistic(x, beta));
end

function testClass
    display 'Class definition is ok';
end

end
end
