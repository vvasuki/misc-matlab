classdef SupervisedKatz

methods(Static = true)

function [score] = katzGeneralized(M, w)
    % wKatzGen computes the generalized Katz measure with weight between two
    % nodes in a social network. 
    %
    % function [score] = wKatzGen(M, w)
    %
    % INPUT:
    %   M       (cell)   adjacency matrices of multiple graphs, size 1 by m.
    %                       Each matrix in M is normalized (Frobenius).
    %   w       (matrix) the weighting matrix for various power of each slice,
    %					 size maxiter by n
    %
    % OUTPUT:
    %   score   (matrix) score matrix, size n by n
    
    if nargin < 2
            error('There must be two input arguments.\n');
    end
    
    score = 0;
    for i = 1:length(M)
            score = score + M{i} * w(i);
    end
    
end % wKatzGen function

function testWeights(Training, validationEdgeSet, beta, numIterations, w)
    % Check to see if w is better at predicting validationEdgeSet than powers of beta.
    
    hybridSources = hybridPower({Training*Training'}, numIterations);
    hybridSources(1) = [];
    for source=1:size(hybridSources,1)
        hybridSources{source} = logical(hybridSources{source} * Training);
    end
    
    display 'Compute the score matrix'
    tic
    Score = wKatzGen(hybridSources, w);
    toc

    % Make predictions based on the Score matrix.
    [scoreVector, sortedIndices] = linkPrediction.Predictor.processSimilarityMatrix(Training, Score, numPredictions);
    
    % Check predictions.
    [precisionSKatz, completenessSKatz] = checkPrediction(scoreVector, sortedIndices, raining, predictedEdgeIndicesSKatz, validationEdgeSet);
    
    v = [];
    for j=2:numIterations
        v = [v; beta^j];
    end
    
    display 'Compute the katz score matrix using the beta vector'
    tic
    katzScoreMatrix = wKatzGen(hybridSources, v);
    toc

    % Make predictions based on the new 'weights'.
    [scoreVector, sortedIndices] = linkPrediction.Predictor.processSimilarityMatrix(Training, katzScoreMatrix, numPredictions);
    
    % Check predictions.
    [precisionKatz, completenessKatz] = checkPrediction(scoreVector, sortedIndices, raining, predictedEdgeIndicesKatz, validationEdgeSet);
    
    % Compare performance.
    comparePerformance(precisionSKatz, completenessSKatz, precisionKatz, completenessKatz);
    
end

function [w] = lsqLP(M,target, method, regPenalty)
    
    % LSQLP learns the weight for the link prediction problem based on least-squares 
    % weighting scheme in the social network. 
    %
    % function [w] = lsqLP(M, target, maxiter)
    %
    % INPUT:
    %   M       (cell) set of adjacency matrices of the graph, size 1 by m
    %           Each matrix is normalized (Frobenius).
    %   target  (vector) the target vector for determing the weights
    %
    % OUTPUT:
    %   w       (matrix) the weighting matrix, size maxiter by m
    %  
    %  Implicitly clears M.
    %  Thanks to Wei Tang.
    
    fprintf(1,'lsqLP: Method : %d \n ', method);
    
    if(method == 1 || method == 3)
        A = zeros(length(target),length(M));
        display '  Creating A' 
        for i = 1:length(M)
            A(:,i) = M{i}(:);
        end
    end
    
    if(method == 1)
        % method 1: nonnegative least squares
        w = lsqnonneg(A,target);
    end
    
    if(method == 2)
        % method 2: normal equation: Solve A'Aw = A' target.
        %  The implementation must not use too much memory.
        display '  calculating X = A^T A, y = A^T target'
        numSources = length(M);
        tic
        X = zeros(numSources, numSources);
        y = zeros(numSources,1);
        for i = 1:numSources 
        a_i = double(M{i}(:));
        for j = 1:numSources 
        a_j = double(M{j}(:));
        X(i,j) = a_i' * a_j;
        end
        y(i) = a_i'*target;
        end
        toc
        w = X \ y;
    end
    
    if(method == 2.5)
        % method 2.5: normal equation with quadratic regularizer: Solve (A'A + lI)w = A' target.
        %  The implementation must not use too much memory.
        display '  calculating X = A^T A, y = A^T target'
        numSources = length(M);
        tic
        X = zeros(numSources, numSources);
        y = zeros(numSources,1);
        for i = 1:numSources 
        a_i = double(M{i}(:));
        for j = 1:numSources 
        a_j = double(M{j}(:));
        X(i,j) = a_i' * a_j;
        end
        y(i) = a_i'*target;
        end
        toc
        w = (X -regPenalty*eye(size(X)))\ y;
    end
    
    if(method == 3)
        % method 3: constrained minimization
        w = LassoGaussSeidel(A,target,regPenalty);
    end
    
end % lsqLP function

end
end