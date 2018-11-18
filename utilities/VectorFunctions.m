classdef VectorFunctions
%  Includes functions which return vectors.
methods(Static=true)
function y = doElementwise(f, x)
    y = zeros(size(x));
    for i = 1:numel(x)
        y(i) = f(x(i));
    end
end

function indicatorVector = getIndicatorVector(x, values)
%      INPUT: x: a scalar, values: a vector
%      OUTPUT: indicatorVector, which has the same dimensions as the values vector.
    numValues = numel(values);
    indicatorVector = (values == x);
end

function IndicatorMatrix = getIndicatorMatrixFromVector(vector, values)
%      INPUT: vector, values
%      OUTPUT: IndicatorMatrix: numel(vector) * numValues matrix.
    vectorLength = numel(vector);
    IndicatorMatrix = false(vectorLength, numel(values));
    for(i = 1: vectorLength)
        IndicatorMatrix(i, :) = VectorFunctions.getIndicatorVector(vector(i), values);
    end
end

function outputVector = getColumnVector(inputVector)
%      Input: a row or a column vector.
%      Output: a column vector, or an error if the input is not a 1 dimensional array.
    [numRows numCols] = size(inputVector);
    assert(numRows ==1 || numCols ==1, 'badInput: Only row or column vectors are accepted by this method.');
    if(numRows ==1)
        outputVector = inputVector';
    else
        outputVector = inputVector;
    end
end

function vectorParts = getRoughlyEqualParts(inputVector, numParts)
    vectorParts = {};
    numElements = numel(inputVector);
    minElementsPerPart = floor(numElements/ numParts);
    numPartsToOverload = numElements - minElementsPerPart*numParts;

    function addParts(elementsPerPart, inputVectorA, numPartsA)
        for(part = 1:numPartsA)
            vectorPart = inputVectorA((part-1)*elementsPerPart+1:min(part*elementsPerPart, numElements));
            if(~isempty(vectorPart))
                vectorParts{end+1} = vectorPart;
            else
                break;
            end
        end
    end
    addParts( minElementsPerPart+1, inputVector, numPartsToOverload);
    addParts( minElementsPerPart, inputVector(numPartsToOverload*(minElementsPerPart+1)+1: end), numParts - numPartsToOverload);
end

function mixedQuadraticFtrVector = mixedQuadraticFeatureMap(x)
% INPUT: x: a row or column vector.
% OUTPUT: mixedQuadraticFtrVector, a vector corresponding to triu(x*x') when x is a column vector.
    x = VectorFunctions.getColumnVector(x);
    X = x*x';
    indicesUpperTriangular = find(triu(ones(size(X)), 1));
    mixedQuadraticFtrVector = X(indicesUpperTriangular);
end

end
end
