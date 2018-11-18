classdef BinaryClassificationTest
properties(Static=true)
    LOG_PATH = [system.SystemUtils.HOME_DIR '/projectsSvn/discriminativeModelUtility/log/'];
end

methods(Static=true)
% nohup-run command:
% echo "probabilisticModels.BinaryClassificationTest.discriminativeVsGenerativeL1RegLogistic(5); exit;">/u/vvasuki//projectsSvn/discriminativeModelUtility/log/tmp/job.in; nohup ssh uvanimor-8 "matlab -nodisplay -nosplash </u/vvasuki//projectsSvn/discriminativeModelUtility/log/tmp/job.in" >/u/vvasuki//projectsSvn/discriminativeModelUtility/log/tmp/job.out 2>/u/vvasuki//projectsSvn/graphicalModels/log/tmp/job.err &

function discriminativeVsGenerativeL1RegLogistic(numNodes, modelsToTest)
    % probabilisticModels.BinaryClassificationTest.discriminativeVsGenerativeL1RegLogistic(5)
    import probabilisticModels.*;
    import statistics.*;
    
    actualModel.topology = 'chain';
    actualModel.Pr_y1 = 0.6;
    actualModel.numNodes = numNodes;
    [actualModel.model_0, actualModel.model_1] = GenerativeModel.makeTestIsingModel(actualModel.numNodes, actualModel.topology);

    experimentSettings.crossValidationFolds = 5;
    experimentSettings.numSamplesTrainingRange = [100:800:2000*log(actualModel.numNodes)];
    % experimentSettings.numSamplesTrainingRange = [100];
    if(~exist('modelsToTest'))
        experimentSettings.modelsToTest = 'disc_gen';
    else
        experimentSettings.modelsToTest = modelsToTest;
    end

    experimentSettings.numSamplesTest = 2*max(experimentSettings.numSamplesTrainingRange);
    experimentSettings.cRangeDiscriminative = [10^(-2)*[1:1:10] 10^(-1)*[2:5]];
    experimentSettings.cRangeGenerative = [10^(-2)*[1:1:10] 10^(-1)*[2:5]];

    % experimentSettings.cRangeDiscriminative = [.1];
    experimentSettings.cRangeGenerative = [.1];

    experimentSettings.numTrials = 5;
    experimentSettings.experimentName = [actualModel.topology num2str(actualModel.numNodes) 'n-' experimentSettings.modelsToTest] ;

    samplingFn = @(numSamplesTraining)GenerativeModel.getSamples(actualModel, numSamplesTraining);
    
    [SamplesTest, yTest] = samplingFn(experimentSettings.numSamplesTest);
    
    % learner corresponding to logisticRegressionl1RegWithValidation
    featureMap = @(SamplesTraining)MatrixFunctions.vectorFnToEachColumn(SamplesTraining, @VectorFunctions.mixedQuadraticFeatureMap);
    % Explicitly adding intercept terms above is not necessary if numel(parameters) = 1 + numel(x).

    % For debugging.
    function misclassificationRateBest = checkMinRisk
        import probabilisticModels.*;
        import statistics.*;
        theta = GenerativeModel.getDiscriminativeModelParameters(actualModel);
        display('Best theta is :');
        theta'
        bestClassifier = Classifier.classifierFromParameters(@LogLinearModels.classifyLogistic, theta);

        X = SamplesTest;
        y = yTest;
        misclassificationRateBest = ClassificationEvaluation.getClassifierMisclassificationRate(bestClassifier, featureMap(X), y)
        learnerTmp = @(X, y)bestClassifier;
        misclassificationRateBestCV = ClassificationEvaluation.getMisclassificationRateFromCrossValidation(featureMap(X), y, learnerTmp, 10)
    end



    function [learnerDiscriminiative, learnerDiscriminiativeUnregularized] = getDiscriminativeClassifiers()
        parameterLearner = @(SamplesTraining, yTraining)LogLinearModels.logisticRegressionl1RegWithValidation(yTraining, featureMap(SamplesTraining), experimentSettings.cRangeDiscriminative, experimentSettings.crossValidationFolds);

        learnerDiscriminiative = @(SamplesTraining, yTraining)Classifier.classifierFromParameters(@LogLinearModels.classifyLogistic, parameterLearner(SamplesTraining, yTraining));

        % theta = parameterLearnerDiscriminative(SamplesTest, yTest);
        % theta'

        parameterLearnerUnregularized = @(SamplesTraining, yTraining)LogLinearModels.logisticRegressionl1Reg(yTraining, featureMap(SamplesTraining), 0);

        learnerDiscriminiativeUnregularized = @(SamplesTraining, yTraining)Classifier.classifierFromParameters(@LogLinearModels.classifyLogistic, parameterLearnerUnregularized(SamplesTraining, yTraining));
    end

    function learnerGenerative = getGenerativeClassifiers()
        import graph.*;
        % actualModel.model_0 = IsingModelLearner.checkModelLearnability(actualModel.model_0, SamplesTest(:, yTest == -1));
        % actualModel.model_1 = IsingModelLearner.checkModelLearnability(actualModel.model_1, SamplesTest(:, yTest == 1));
        % cCalculated = 1* max(actualModel.model_0.c, actualModel.model_1.c);

        % conditionalModelLearner = @(Samples, model)IsingModelLearner.isingModelFromSamplesAutopickLambda(Samples, experimentSettings.cRangeGenerative, model);

        % conditionalModelLearner = @(Samples, model)IsingModelLearner.isingModelFromSamplesAutopickLambda(Samples, experimentSettings.cRangeGenerative, experimentSettings.crossValidationFolds, model);

        conditionalModelLearner = @(Samples, model)IsingModelLearner.treeModelFromSamples(Samples);

        generativeModelParameterLearner = @(Samples, y)GenerativeModel.getDiscriminativeModelParameters(GenerativeModel.learnModel(Samples, y, conditionalModelLearner, actualModel))
        % keyboard
        learnerGenerative = @(SamplesTraining, yTraining)Classifier.classifierFromParameters(@LogLinearModels.classifyLogistic, generativeModelParameterLearner(SamplesTraining, yTraining));
    end


    SamplesTestMapped = featureMap(SamplesTest);
    misclassificationRateBest = checkMinRisk();
    errorLeast = misclassificationRateBest*ones(size(experimentSettings.numSamplesTrainingRange))
    % keyboard

    yData = {errorLeast};
    legendNames = {'Min Error'};
    if(StringUtilities.isSubstring(experimentSettings.modelsToTest, 'gen'))
        learnerGenerative = getGenerativeClassifiers();
        errorGenerative = ClassificationEvaluation.getSampleComplexity(learnerGenerative, experimentSettings.numSamplesTrainingRange, samplingFn, SamplesTestMapped, yTest, experimentSettings.numTrials);

        yData{end+1} = errorGenerative;
        legendNames{end+1} = 'Gen';
    end
    if(StringUtilities.isSubstring(experimentSettings.modelsToTest, 'disc'))
        [learnerDiscriminiative, learnerDiscriminiativeUnregularized] = getDiscriminativeClassifiers();

        errorDiscriminative = ClassificationEvaluation.getSampleComplexity(learnerDiscriminiative, experimentSettings.numSamplesTrainingRange, samplingFn, SamplesTestMapped, yTest, experimentSettings.numTrials);

        fprintf('Trying to learn unregularized.\n');
        errorDiscriminativeUnregularized = ClassificationEvaluation.getSampleComplexity(learnerDiscriminiativeUnregularized, experimentSettings.numSamplesTrainingRange, samplingFn, SamplesTestMapped, yTest, experimentSettings.numTrials);

        yData{end+1} = errorDiscriminative;
        legendNames{end+1} = 'Discr';
        yData{end+1} = errorDiscriminativeUnregularized;
        legendNames{end+1} = 'DiscrUnreg';
    end
    
    [figureHandle, fullFileNameSansExtension] = IO.plotAndSave(experimentSettings.numSamplesTrainingRange, yData, 'numSamples', 'Misclassification rate',  BinaryClassificationTest.LOG_PATH, experimentSettings.experimentName, '', legendNames);

    save([fullFileNameSansExtension '.mat'], 'actualModel', 'experimentSettings', '-append');

end

function testClass
    display 'Class definition is ok';
end

end
end
