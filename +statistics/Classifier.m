classdef Classifier
methods(Static=true)
function classifier = classifierFromParameters(functionFamily, parameters)
    classifier = @(x)functionFamily(x, parameters);
end

function testClassifierFromParameters
    import probabilisticModels.*;
    import statistics.*;
    
    functionFamily = @(x, beta)LogLinearModels.classifyLogistic(x, beta);
    parameters = [1 2];
    classifier = Classifier.classifierFromParameters(functionFamily, parameters);
    fprintf('%s \n', func2str(classifier));
    fprintf('%d \n', classifier([2 -30]));
end

function testClass()
    display 'Class is ok!'
end

end
end