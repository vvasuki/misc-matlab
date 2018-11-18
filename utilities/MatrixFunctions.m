classdef MatrixFunctions
% Also see the class MatrixTransformers.
methods(Static=true)
function [sumAlongRows, sumAlongColumns] = getRowColSums(A)
    sumAlongRows = sum(A, 1);
    sumAlongColumns = sum(A, 2);
end

function [sumGraphs] = plotRowColSums(A)
    sumGraphs = figure;
    [sumAlongRows, sumAlongColumns] = MatrixFunctions.getRowColSums(A);
    subplot(2,1,1);
    grid on;
    title('Sum along rows');
    bar(sumAlongRows);

    subplot(2,1,2);
    grid on;
    title('Sum along columns');
    bar(sumAlongColumns);
end

function [avgSumAlongRows, avgSumAlongCols, minSumAlongRows, minSumAlongCols, maxSumAlongRows, maxSumAlongCols, numZeroRows, numZeroCols] = getRowColSumStatistics(A)
    [numRows, numCols] = size(A);
    [sumAlongRows, sumAlongColumns] = MatrixFunctions.getRowColSums(A);
    avgSumAlongRows = full(sum(sumAlongRows))/numCols; avgSumAlongCols = full(sum(sumAlongColumns))/numRows;
    minSumAlongRows = full(min(sumAlongRows)); minSumAlongCols = full(min(sumAlongColumns));
    maxSumAlongRows = full(max(sumAlongRows)); maxSumAlongCols = full(max(sumAlongColumns));
    numZeroRows = numRows - nnz(sumAlongColumns);
    numZeroCols = numCols - nnz(sumAlongRows);
end

function density = getDensity(A)
    [numRows, numCols] = size(A);
    density = nnz(A)/(numRows*numCols);
end

function sparsity = getSparsity(A)
    sparsity = 1 - MatrixFunctions.getDensity(A);
end

function bPositiveDefinite = positiveDefinitenessChecker(Z)
    [Z_c bNotPositiveDefinite] = chol(Z);
    bPositiveDefinite = ~bNotPositiveDefinite;
end

function resultVector = functionalToEachRow(Matrix, fn)
    [numRows, numCols] = size(Matrix);
    resultVector = zeros(numRows, 1);
    for(row=1:numRows)
        resultVector(row) = fn(Matrix(row, :));
    end
end

function [resultVector, paramsVector ] = functionalToEachColumn(Matrix, fn, initParamsForWarmStart)
    [numRows, numCols] = size(Matrix);
    resultVector = zeros(numCols, 1);
    paramsVector = {};
    if(exist('initParamsForWarmStart'))
        bWarmStartFn = true;
    else
        bWarmStartFn = false;
    end
    for(col=1:numCols)
        if(bWarmStartFn)
            [resultVector(col), initParamsForWarmStart] = fn(Matrix(:, col), initParamsForWarmStart);
            paramsVector{col} = initParamsForWarmStart;
        else
            resultVector(col) = fn(Matrix(:, col));
        end
    end
end

function Result = vectorFnToEachColumn(Matrix, fn)
    [numRows, numCols] = size(Matrix);
    colArray = mat2cell(Matrix, numRows, ones(1, numCols));
    resultArray = CellFunctions.applyFunctionToAllCells(fn, colArray);
    Result = cell2mat(resultArray);
end

function [bIsSymmetric maxImbalance] = isSymmetric(A)
    [m, n] = size(A);
    maxImbalance = Inf;
    B = abs(A - A');
    bIsSymmetric = (m==n) && (~any(B(:)));
    if(m==n)
        maxImbalance = max(B(:));
    end
end

function A = balanceSymmetricMatrixDiagonalAddition(A)
    assert(MatrixFunctions.isSymmetric(A), 'Only works with symmetric matrices');
    colSums = sum(abs(A), 2);
    maxColSum = max(colSums);
    A = A + diag(maxColSum - colSums);
    A = A/maxColSum;
end

function A = balanceSymmetricMatrixSinkhorn(A, maxNumIterations, assymetryTolerance)
    assert(MatrixFunctions.isSymmetric(A), 'Only works with symmetric matrices');
    fprintf('balanceSymmetricMatrixSinkhorn\n');
    for(i=1:maxNumIterations)
        % normalize rows
        A = normr(A);

        % normalize columns
        A = normc(A);
        [bIsSymmetric maxImbalance] = MatrixFunctions.isSymmetric(A);
        if(maxImbalance< assymetryTolerance)
            fprintf('Done! numIterations: %d\n', i);
            break;
        end
    end
end

function testBalanceSymmetricMatrix()
    A = rand(4);
    A = A + A';
    A = A - diag(diag(A));
    MatrixFunctions.isSymmetric(A)
    maxNumIterations = Inf;

    B = MatrixFunctions.balanceSymmetricMatrixSinkhorn(A, maxNumIterations, 10^-3);
    MatrixFunctions.isSymmetric(B)

    C = MatrixFunctions.balanceSymmetricMatrixDiagonalAddition(A);
    MatrixFunctions.isSymmetric(C)
end


end
end
