classdef ExponentialFamily
methods(Static=true)
function cumulativeDensity = normalCumulativeDensity(x, expectation, stdDeviation)
%   Let X \distr N(expectation, stdDeviation^2). Then what is CDF at x?
    xShiftedScaled = (x-expectation)/stdDeviation;
    cumulativeDensity = erfc(-xShiftedScaled/sqrt(2))/2;
end

function x = invNormalCumulativeDensity(cumulativeDensity, expectation, stdDeviation)
	xShiftedScaled = -sqrt(2)*erfcinv(2*cumulativeDensity);
    x = xShiftedScaled*stdDeviation + expectation;
end



function testNormalCumulativeDensity
	import probabilisticModels.*;
	expectation = 0;
	stdDeviation = 1;
	
	xActual = 0;
	cumulativeDensity = ExponentialFamily.normalCumulativeDensity(xActual, expectation, stdDeviation);
	x = ExponentialFamily.invNormalCumulativeDensity(cumulativeDensity, expectation, stdDeviation);
	fprintf('cumulativeDensity: %d, inverse: %d, expected: 0.5, %d \n', cumulativeDensity, x, xActual);

	xActual = Inf;
	cumulativeDensity = ExponentialFamily.normalCumulativeDensity(xActual, expectation, stdDeviation);
	x = ExponentialFamily.invNormalCumulativeDensity(cumulativeDensity, expectation, stdDeviation);
	fprintf('cumulativeDensity: %d, inverse: %d, expected: 1, %d \n', cumulativeDensity, x, xActual);

	xActual = -Inf;
	cumulativeDensity = ExponentialFamily.normalCumulativeDensity(xActual, expectation, stdDeviation);
	x = ExponentialFamily.invNormalCumulativeDensity(cumulativeDensity, expectation, stdDeviation);
	fprintf('cumulativeDensity: %d, inverse: %d, expected: 0, %d \n', cumulativeDensity, x, xActual);
end

function testClass
    display 'Class definition is ok';
end

end
end
