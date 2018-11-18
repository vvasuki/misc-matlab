classdef Functionals
methods(Static=true)
function [output] = extendedValueFunctionalWrapper(functionHandle, upperBound, lowerBound, x)
%      Input:
%          functionHandle : the function which is being wrapped.
%          upperBound : A vector.
%          lowerBound : A vector.
%      Output:
%          output: The output of an extended value functional corresponding to functionHandle, which follows the specification of functionHandle between lower and upper bounds, but is Inf outside it.
    if(all(x <= upperBound) && all(x >= lowerBound))
        output = functionHandle(x);
    else
        output = Inf;
    end
end

function [output] = linearApproxBetweenIntervals(functionHandle, domain, x)
%      Input:
%          functionHandle : the extended Value function which is being wrapped.
%          domain : a cell array of matrices. Each matrix has this form:
%          n \times 2 matrix of lower bounds and upper bounds of matrices.
%      Output:
%          output: The output of a functional which behaves exactly like functionHandle, where it is defined; but behaves like a piecewise-linear approximation where it is not defined. 
%      ASSUMPTION: domains cannot contain +Inf or -Inf.
    numDimensions = length(x);
    x_l = x;
    x_u = x;
    nonDomainDimension = 0;
    for i=1:numDimensions
        [x_l(i), x_u(i)] = functionals.Functionals.getClosestPointsInDomain(domain{i}, x(i));
        if(x_l(i) ~= x_u(i))
            nonDomainDimension = i;
            break;
        end
    end
    if(nonDomainDimension == 0)
        output = functionHandle(x);
    else
        alpha = x(nonDomainDimension) - x_l(nonDomainDimension);
        if(alpha == Inf || isnan(alpha))
            output = Inf;
        else
             f_x_l = functionals.Functionals.linearApproxBetweenIntervals(functionHandle, domain, x_l);
             f_x_u = functionals.Functionals.linearApproxBetweenIntervals(functionHandle, domain, x_u);
             output = f_x_l + alpha*(f_x_u - f_x_l) / (x_u(nonDomainDimension) - x_l(nonDomainDimension));
        end
    end
end

function [x_l, x_u] = getClosestPointsInDomain(Domain, x)
%      Input: 
%          domain : 
%             n \times 2 matrix of lower bounds and upper bounds of matrices.
%          x : a scalar
%      Output:
%          x_l: argmin_{y \in Domain, y \leq x} x-y, 
%          x_u: argmin_{y \in Domain, y \geq x} y-x
    if(functionals.Functionals.checkDomainValidity(Domain) == false)
        error('Domain invalid!');
        Domain
    end
    Domain = Domain';
    x_l_index = find(Domain <= x, 1, 'last');
    if(numel(x_l_index) == 0)
        x_l = -Inf;
    else
        x_l = Domain(x_l_index(1));
    end
    x_u_index = find(Domain >= x, 1, 'first');
    if(numel(x_u_index) == 0)
        x_u = Inf;
    else
        x_u = Domain(x_u_index(1));
    end
    
    if(numel(x_l_index)>0 && numel(x_u_index)>0 )
        if(x_l_index <= x_u_index - 1 && mod(x_u_index, 2) == 0)
            x_l = x;
            x_u = x;
        end
    end
%          keyboard
end

function domainValidity = checkDomainValidity(Domain)
    domainValidity = all(Domain(:, 1) <= Domain(:, 2));
end

function testGetClosestPointsInDomain()
    Domain = [1 2; 3 4; 5 5];
    x = 2.5;
    [x_l, x_u] = functionals.Functionals.getClosestPointsInDomain(Domain, x)
    fprintf('Expected: 2, 3\n');
    x = 1.5;
    [x_l, x_u] = functionals.Functionals.getClosestPointsInDomain(Domain, x)
    fprintf('Expected: 1.5, 1.5\n');
    x = 5;
    [x_l, x_u] = functionals.Functionals.getClosestPointsInDomain(Domain, x)
    fprintf('Expected: 5, 5\n');
    x = 6;
    [x_l, x_u] = functionals.Functionals.getClosestPointsInDomain(Domain, x)
    fprintf('Expected: 5, Inf\n');
    x = -1;
    [x_l, x_u] = functionals.Functionals.getClosestPointsInDomain(Domain, x)
    fprintf('Expected: -Inf, 1\n');
end

function domainSetsAltered = fixVariableInDomainSets(domainSets, variableToFix, value)
%  Fix variable in a set of discrete domain sets to a certain value.
    domainSetsAltered = domainSets;
    domainSetsAltered{variableToFix} = [value];
end


function testLinearApproxBetweenIntervals()
    Domain1 = [1 2; 3 4; 5 5];
    Domain2 = [2 2; 3 4; 5 6];
    domain = {Domain1, Domain2};
    functionHandle = @(x)ones(length(x),1)'*x;
    lowerBound = min([Domain1(:)'; Domain2(:)'], [], 2);
    upperBound = max([Domain1(:)'; Domain2(:)'], [], 2);
    extendedValueFnHandle = @(x)functionals.Functionals.extendedValueFunctionalWrapper(functionHandle, upperBound, lowerBound, x);
    
    display 'Testing extendedValueFunctionalWrapper!';
    x = [55; 0];
    output = extendedValueFnHandle(x);
    fprintf('Actual output: %d, Expected: %d\n', output, Inf);
    
    x = [-Inf; 2];
    output = extendedValueFnHandle(x);
    fprintf('Actual output: %d, Expected: %d\n', output, Inf);
    
    
    display 'Testing linearApproxBetweenIntervals!';
    linearApproxBetweenIntervalsFn = @(x)functionals.Functionals.linearApproxBetweenIntervals(extendedValueFnHandle, domain, x);
    x = [2; 2];    
    [output] = linearApproxBetweenIntervalsFn(x);
    fprintf('Actual output: %d, Expected: %d\n', output, 4);

    x = [-50; 2];    
    [output] = linearApproxBetweenIntervalsFn(x);
    fprintf('Actual output: %d, Expected: %d\n', output, Inf);

    x = [2; 2.3];    
    [output] = linearApproxBetweenIntervalsFn(x);
    fprintf('Actual output: %d, Expected: %d\n', output, 4.3);
    
    x = [2.5; 2.3];    
    [output] = linearApproxBetweenIntervalsFn(x);
    fprintf('Actual output: %d, Expected: %d\n', output, 4.3+0.5);

    x = [2.6; 3.5];
    [output] = linearApproxBetweenIntervalsFn(x);
    fprintf('Actual output: %d, Expected: %d\n', output, 5.5 + 0.6*1);

end

function testClass
    display 'This class is OK!';
end

end
end