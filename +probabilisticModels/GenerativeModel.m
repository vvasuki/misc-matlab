classdef GenerativeModel
properties(Static=true)
end
methods(Static=true)
function [model_0, model_1] = makeTestIsingModel(numNodes, topology)
    import probabilisticModels.*;
    
    degmax = ceil(0.1 * numNodes);
    couplingStrength = min(0.1,2.5/degmax);
    % couplingType = 'mixed';
    couplingType = 'positive';
    
    [model_0] = DiscreteRVGraphicalModel.getBinaryRVGraphicalModel(topology, numNodes, degmax, couplingStrength, couplingType);
    
    model_1 = DiscreteRVGraphicalModel.negateModel(model_0);
end

function [Samples, y] = getSamples(actualModel, numSamples)
    import probabilisticModels.*;
    import graph.*;
    Pr_y1 = actualModel.Pr_y1;

    Pr_y0 = 1- Pr_y1;

    numY1Samples = ceil(numSamples*Pr_y1);
    Samples_1 = DiscreteRVGraphicalModel.getSamples(actualModel.model_1, numY1Samples);
    
    Samples_0 = DiscreteRVGraphicalModel.getSamples(actualModel.model_0, numSamples - numY1Samples);
    
    Samples = [Samples_0 Samples_1];
    Samples(Samples == 2) = -1;
    y = [-1*ones(1, numSamples - numY1Samples) ones(1, numY1Samples) ];
end

function theta = getDiscriminativeModelParameters(model)
    model_0 = model.model_0;
    model_1 = model.model_1;
    Pr_y1 = model.Pr_y1;

    % Indices of entries in model_i.EdgeParameters which are used to construct the discriminative classifier. Also useful to count the number of such enties.
    thetaIndices = logical(triu(ones(size(model_0.EdgeParameters)), 1));

    if(~(model_0.bIsTree || model_1.bIsTree ))
        theta = zeros(sum(thetaIndices(:))+ 1, 1);
        display('Got non-tree model');
        return;
    end
    assert(model_0.bIsTree && model_1.bIsTree , 'Ising model learned is not a tree!');
    
    theta_0 = log(Pr_y1 / (1 - Pr_y1)) - model_1.logPartitionFn + model_0.logPartitionFn;
    % model_0.EdgeParameters'
    % model_1.EdgeParameters'
    thetaTmp = model_1.EdgeParameters - model_0.EdgeParameters;
    theta = [theta_0; thetaTmp(thetaIndices)];
end

function model = learnModel(Samples, y, conditionalModelLearner, actualModel)
    % Separate the samples.
    positiveSampleIndices = (y == 1);
    negativeSampleIndices = ~positiveSampleIndices;
    Samples_0 = Samples(:, negativeSampleIndices);
    Samples_1 = Samples(:, positiveSampleIndices);

    % Estimate Pr_y1.
    model.Pr_y1 = sum(positiveSampleIndices)/ numel(y);

    % Learn model_0 and model_1.
    model.model_0 = conditionalModelLearner(Samples_0, actualModel.model_0);
    model.model_1 = conditionalModelLearner(Samples_1, actualModel.model_1);
end

function testClass
    display 'Class definition is ok';
end

end
end
