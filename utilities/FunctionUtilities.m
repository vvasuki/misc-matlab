classdef FunctionUtilities

methods(Static=true)
function [resultA resultB] = combine2functions(fnA, fnB, input)
    resultA = fnA(input);
    resultB = fnB(input);
end


function testClass
    display 'Class definition is ok';
end

end
end
