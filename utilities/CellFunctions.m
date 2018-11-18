classdef CellFunctions

methods(Static=true)

function resultArray = applyFunctionToAllCells(fn, cellArray)
    % INPUT:
    %  fn: function handle.
    %  cellArray: the operands.
    % OUTPUT:
    %  resultArray: a cell array of results.
    % cellFun only applies functionals to all cells, unlike this function.
    cellArraySize = size(cellArray);
    resultArray = cell(cellArraySize);
    for i=1:prod(cellArraySize)
        resultArray{i} = fn(cellArray{i});
    end
end


function testClass
    display 'Class definition is ok';
end

end
end
