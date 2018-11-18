classdef MatrixTransformer
%  A bunch of methods related to MatrixTransformer operations on or transformations of matreces.
methods(Static=true)
function [PatternMatrix] = getPatternMatrix(DataMatrix, numRows, numCols)
    % An extension of spconvert which adds some zero rows and zero columns.
    % DataMatrix: a matrix where the first column lists row DataMatrix and column 2 indicates column DataMatrix
    % of the 1's in the sparse matrix.
    [rows, cols]=size(DataMatrix);
    if cols == 2
        DataMatrix(:,3) = ones(rows,1);
    end
    if(sum(DataMatrix(:,1) == numRows)==0 || sum(DataMatrix(:,2) == numCols)==0)
        DataMatrix(end+1, :) = [numRows, numCols, 0];
    end
    PatternMatrix = spconvert(DataMatrix);
end

function outputMatrix = permuteMatrix(inputMatrix, rowOrder, columnOrder)
%      Permute rows and columns of inputMatrix according to order specified.
    PR = MatrixTransformer.makePermutationMatrix(rowOrder);
    PR = PR';
    PC = MatrixTransformer.makePermutationMatrix(columnOrder);
    outputMatrix = PR'*inputMatrix*PC;
end

function [P] = makePermutationMatrix(columnOrder)
%      Columns in I are e(:,i).
%      Input: Order of e(:,i), as they should appear in P.
%      Output: P.
%      Example: Input [2,1] should yield [0 1; 1 0].
%  Ceveat: no error handling implemented to check if the output is a valid permutation matrix: columns or rows could be repeated.
    columnOrder = MatrixTransformer.getColumnVector(columnOrder);
    sizeP = size(columnOrder, 1);
    A(:,1) = columnOrder;
    A(:,2) = [1:sizeP]';
    A(:,3) = ones(sizeP,1);
    P = spconvert(A);
end

function A = sparsify(A, density)
%      Input: Matrix A, density.
%      Output: Matrix A with needed density.

%      Calculate the number of elements to keep.
    % display('Function: sparsify');
    [numRows, numCols] = size(A);
    if(density>=1)
        return;
    end
    
    numToRetain = floor(density* numRows* numCols);

    A = MatrixTransformer.retainTopEntries(A, numToRetain);
end

function A = retainTopEntries(A, numToRetain)
    [numRows, numCols] = size(A);
    if(numToRetain > numRows*numCols)
        return;
    end

    %display('  sorting');
    [sortedElements, indices] = sort(full(A(:)), 'descend');
    % fprintf(1, '  retaining %d nonZeroes \n', numToRetain);
    A = zeros(numRows, numCols);
    indices = indices(1:numToRetain);
    sortedElements = sortedElements(1:numToRetain);
    % display('  Preparing A!');
    A(indices) = sortedElements;
    A = sparse(A);
    % display('  done!');
end

end
end