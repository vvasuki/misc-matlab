classdef PerformanceTester
methods(Static=true)
    
    function [avgPrecisions, avgSensitivities, avgSpecificities] = test(predictor, datasetName, bGetNewSplits, initParams, bTestMultipleAffiliationRanges, affRangeQuantiles, bBalanceUserUser)
    % Input: predictor: a function handler
        import affiliationRecommendation.*;
        fracEdgesToRemove = 0.3;
        fracEdgesToRemoveValidation = 0.1;
        if(~exist('initParams'))
            initParams = struct();
        end
        
        [UserUser, Training, TargetEdgeSet, TrainingValid, ValidationEdgeSet, clusters, remappingKeyForClustering] = affiliationRecommendation.Data.getExperimentData(datasetName, bGetNewSplits, fracEdgesToRemove, fracEdgesToRemoveValidation);
        initParams.clusters = clusters;
        initParams.remappingKeyForClustering = remappingKeyForClustering;
        initParams.datasetName = datasetName;
        
        whos
        if(exist('bBalanceUserUser') && bBalanceUserUser)
            % Takes very long
            % UserUser = MatrixFunctions.balanceSymmetricMatrixSinkhorn(UserUser, Inf, 10^-1);
            
            UserUser = MatrixFunctions.balanceSymmetricMatrixDiagonalAddition(double(UserUser));
        end
        
        if (~exist('bTestMultipleAffiliationRanges') || isempty(bTestMultipleAffiliationRanges) || ~bTestMultipleAffiliationRanges)
            affiliationRangeUpperBounds = Inf;
            [avgPrecision, avgSensitivity, avgSpecificity] = PerformanceTester.testAffiliationRange(predictor, UserUser, Training, TargetEdgeSet, TrainingValid, ValidationEdgeSet, [], [], datasetName, initParams);
        else
            numAffiliations = sum(Training, 2);
            affiliationRangeUpperBounds = prctile(numAffiliations, affRangeQuantiles);
            for i=1:numel(affiliationRangeUpperBounds)-1
                affiliationRangeLB = affiliationRangeUpperBounds(i);
                affRangeQuantileLB = affRangeQuantiles(i);
                affiliationRange = [affiliationRangeLB affiliationRangeUpperBounds(i+1)];
                fprintf('Considering affiliation range %d...\n',  affiliationRangeUpperBounds(i+1));
                affRangeStr = [num2str(affRangeQuantileLB) '-' num2str(affRangeQuantiles(i+1))];
                [avgPrecision, avgSensitivity, avgSpecificity] = PerformanceTester.testAffiliationRange(predictor, UserUser, Training, TargetEdgeSet, TrainingValid, ValidationEdgeSet, affiliationRange, affRangeStr, datasetName, initParams);
            end
        end
        display 'All done  - ready for inspection!'
        keyboard
    end
    
    function [avgPrecisions, avgSensitivities, avgSpecificities] = testAffiliationRange(predictor, UserUser, Training, TargetEdgeSet, TrainingValid, ValidationEdgeSet, affiliationRange, affRangeStr, datasetName, initParams)
        import randomizationUtils.*;
        import affiliationRecommendation.*;
        numValidation = 3;
        maxNumPredictions = 50;
        step = 5;
        numPredictionsVector = [step:step:maxNumPredictions];
        
        % Process affiliationRange and TargetEdgeSet
        if(isempty(affiliationRange))
            [numUsers, numGroups] = size(Training);
            affRangeStr = '';
        else
            if(numel(affiliationRange) ~= 2 || affiliationRange(2) < affiliationRange(1))
                error('affiliationRange invalid');
            end
            numAffiliations = sum(Training, 2);
            bUserInRange = ((numAffiliations >= affiliationRange(1)) & (numAffiliations <= affiliationRange(2)));
            TargetEdgeSet(bUserInRange, :) = 0;
            numUsers = full(sum(bUserInRange));
        end
        numPredictionsForValidation = numUsers * maxNumPredictions/2;
        
        predictorName = func2str(predictor);
        timer = Timer();
        ScoreMatrix = predictor(UserUser, Training, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams);
        timer = timer.endTimer();
        if(StringUtilities.isSubstring(predictorName,'Cluster'))
            numClusters = length(initParams.clusters);
            predictorName = [predictorName '-' num2str(numClusters) 'cl'];
        end
        if(isfield(initParams, 'numFactors'))
            predictorName = [predictorName  '-' num2str(initParams.numFactors) 'f-'];
        end
        
        % usersToEvaluate = Sample.sampleWithoutReplacement(, sampleSize);
        datasetName = regexprep(datasetName, 'Cluster.*', '');
        % Ensure that the score matrix is dense: else, evaluation takes a lot of time.
        fprintf('Got score matrix from %s!\n', predictorName);
        
        timingLogFile = fopen([affiliationRecommendation.Constants.LOG_PATH datasetName '/' Constants.TIMING_LOG_FILE], 'a');
        machineName = IO.getHostname();
        fprintf(timingLogFile, 'datasetName: %s\n', datasetName);
        fprintf(timingLogFile, 'Predictor: %s Machine: %s \n', predictorName, machineName);
        fprintf(timingLogFile, 'Time: %d \n', timer.elapsedTime);
        fclose(timingLogFile);
        
        
        [avgPrecisions, avgSensitivities, avgSpecificities] = statistics.ClassificationEvaluation.getAveragePerformance(ScoreMatrix, Training, TargetEdgeSet, numPredictionsVector);
        display 'Ready to get the figure'
        figureHandle = IO.plotAndSave((1-avgSpecificities), avgSensitivities, '1-avg. Specificity', 'avg. Sensitivity', [affiliationRecommendation.Constants.LOG_PATH datasetName '/'], [predictorName affRangeStr]);
    end
    
    function experimentRandom(datasetName)
        bGetNewSplits = 0;
        predictor = @affiliationRecommendation.Predictors.randomPredictor;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, []);
    end
    
    function experimentSVD(datasetName)
        bGetNewSplits = 0;
        predictor = @affiliationRecommendation.SVDLatentFactors.getSVDApprox;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, []);
    end
    
    function experimentJointSVDAsymmetricApprox(datasetName)
        bGetNewSplits = 0;
        predictor = @affiliationRecommendation.SVDLatentFactors.jointSVDAsymmetricApprox;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, []);
    end
    
    function experimentJointSVDSymmetricApprox(datasetName, bTestMultipleAffiliationRanges)
        bGetNewSplits = 0;
        affRangeQuantiles = [80 90];
        bBalanceUserUser = 0;
        predictor = @affiliationRecommendation.SVDLatentFactors.jointSVDSymmetricApprox;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, [], bTestMultipleAffiliationRanges, affRangeQuantiles, bBalanceUserUser);
    end
    
    function experimentJointClusteredSVDSymmetricApprox(datasetName)
        bGetNewSplits = 0;
        bBalanceUserUser = 0;
        predictor = @affiliationRecommendation.SVDLatentFactors.jointClusteredSVDSymmetricApprox;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits);
    end
    

    function experimentJoint3SVDSymmetricApprox(datasetName)
        bGetNewSplits = 0;
        predictor = @affiliationRecommendation.SVDLatentFactors.joint3SVDSymmetricApprox;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, []);
    end

    function experimentJoint2SVDSymmetricApprox(datasetName)
        bGetNewSplits = 0;
        predictor = @affiliationRecommendation.SVDLatentFactors.joint2SVDSymmetricApprox;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, []);
    end

    function experimentJoint4SVDSymmetricApprox(datasetName)
        bGetNewSplits = 0;
        predictor = @affiliationRecommendation.SVDLatentFactors.joint4SVDSymmetricApprox;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, []);
    end
    
    function experimentJoint5SVDSymmetricApprox(datasetName)
        bGetNewSplits = 0;
        predictor = @affiliationRecommendation.SVDLatentFactors.joint5SVDSymmetricApprox;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, []);
    end
    
    function experimentJoint6SVDSymmetricApprox(datasetName)
        bGetNewSplits = 0;
        predictor = @affiliationRecommendation.SVDLatentFactors.joint6SVDSymmetricApprox;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, []);
    end
    
    function experimentKatzApprox(datasetName)
        bGetNewSplits = 0;
        predictor = @affiliationRecommendation.Katz.getKatzApprox;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, []);
    end
    
    function experimentJointSVDKatz(datasetName, numFactors)
        bGetNewSplits = 0;
        initParams.numFactors = numFactors;
        predictor = @affiliationRecommendation.SVDLatentFactors.jointSVDKatz;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, initParams);
    end
    
    function experimentJointSVDKatz2(datasetName, numFactors)
        bGetNewSplits = 0;
        initParams.numFactors = numFactors;
        predictor = @affiliationRecommendation.SVDLatentFactors.jointSVDKatz2;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, initParams);
    end
    
    
    
    function experimentJointSVDKatzClustered(datasetName, numFactors)
        bGetNewSplits = 0;
        predictor = @affiliationRecommendation.SVDLatentFactors.jointSVDKatzClustered;
        initParams.numFactors = numFactors;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, initParams);
    end
    
    
    
    function experimentKatzJoint3Approx(datasetName, bTestMultipleAffiliationRanges)
        bGetNewSplits = 0;
        affRangeQuantiles = [80 90];
        predictor = @affiliationRecommendation.Katz.getKatzJoint3Approx;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, [], bTestMultipleAffiliationRanges, affRangeQuantiles);
    end
    
    function experimentKatzJoint3ApproxFminunc(datasetName)
        bGetNewSplits = 0;
        predictor = @affiliationRecommendation.Katz.getKatzJoint3ApproxFminunc;
        affiliationRecommendation.PerformanceTester.test(predictor, datasetName, bGetNewSplits, []);
    end

    function summarizePerformance(datasetName)
    % Produces a figure in our paper.
        import affiliationRecommendation.*;
        logPath = Constants.getLogPath(datasetName);
        summaryPath = Constants.getSummaryPath(datasetName);
        legendNames = {};
        
        filesToCombine = {};
        files = IO.listFiles([logPath 'aff*KatzApprox*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatz(A)';
        
        files = IO.listFiles([logPath 'aff*KatzJoint3Approx20*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatz(C)';
        
        files = IO.listFiles([logPath 'aff*SVDApprox*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'SVD(A)';
        
        files = IO.listFiles([logPath 'aff*jointSVDSymm*Approx20*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'SVD(C)';
        
        figureName = 'summary';
        yLabel = 'avg. Sensitivity';
        xLabel = '1-avg. Specificity';
        figureHandle = IO.getFiguresAndCombine(filesToCombine, summaryPath, figureName, xLabel, yLabel, legendNames);
    end
    
    function summarizePerformanceSVDSymmetric(datasetName)
    % Produces the latent factors predictors figure in our paper.
        import affiliationRecommendation.*;
        logPath = Constants.getLogPath(datasetName);
        summaryPath = Constants.getSummaryPath(datasetName);
        legendNames = {};
        
        filesToCombine = {};
        files = IO.listFiles([logPath 'aff*jointSVDSymm*Approx20*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'SVD(C)';
        
        files = IO.listFiles([logPath 'aff*joint3SVDSymm*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'SVD(C''(D_2))';
        
        files = IO.listFiles([logPath 'aff*joint3SVDSymm*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'SVD(C''(D_3))';
        
        
        figureName = 'summarySVD';
        yLabel = 'avg. Sensitivity';
        xLabel = '1-avg. Specificity';
        figureHandle = IO.getFiguresAndCombine(filesToCombine, summaryPath, figureName, xLabel, yLabel, legendNames);
    end
    
    function summarizePerformanceScalability(datasetName)
    % Produces a figure in our paper.
        import affiliationRecommendation.*;
        logPath = Constants.getLogPath(datasetName);
        summaryPath = Constants.getSummaryPath(datasetName);
        legendNames = {};
        
        filesToCombine = {};
        files = IO.listFiles([logPath 'aff*KatzJoint3Approx20*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatz(C)';
        
        files = IO.listFiles([logPath 'aff*KatzApprox20*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatz(A)';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatz-300f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzLFM(C, 300))';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatz2-300f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzCS(300)';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatzClustered-5cl-300f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzLFM-c(C, 5, 300))';
        
        figureName = 'summaryScalability';
        yLabel = 'avg. Sensitivity';
        xLabel = '1-avg. Specificity';
        figureHandle = IO.getFiguresAndCombine(filesToCombine, summaryPath, figureName, xLabel, yLabel, legendNames);
    end
    
    function summarizeSVDKatz(datasetName)
    % Produces a figure in our paper.
        import affiliationRecommendation.*;
        logPath = Constants.getLogPath(datasetName);
        summaryPath = Constants.getSummaryPath(datasetName);
        legendNames = {};
        
        filesToCombine = {};
        files = IO.listFiles([logPath 'aff*jointSVDSymm*Approx20*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'LFM(C)';
        
        files = IO.listFiles([logPath 'aff*jointClusteredSVDSymmetricApprox*5cl*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'LFM-c(C, 5)';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatz-300f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzLFM(C, 300))';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatz2-300f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzCS(300)';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatzClustered-5cl-300f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzLFM-c(C, 5, 300)';
        
        figureName = 'summarySVDKatz';
        yLabel = 'avg. Sensitivity';
        xLabel = '1-avg. Specificity';
        figureHandle = IO.getFiguresAndCombine(filesToCombine, summaryPath, figureName, xLabel, yLabel, legendNames);
    end
    
    function summarizeClusteredMethodsClusterDependency(datasetName)
    % Produces a figure in our paper.
        import affiliationRecommendation.*;
        logPath = Constants.getLogPath(datasetName);
        summaryPath = Constants.getSummaryPath(datasetName);
        legendNames = {};
        
        filesToCombine = {};
        files = IO.listFiles([logPath 'aff*jointSVDKatzClustered-5cl-300f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzLFM-c(C, 5, 300)';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatzClustered-10cl-300f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzLFM-c(C, 10, 300)';
        
        figureName = 'summaryClusteredMethodsClusterDependency';
        yLabel = 'avg. Sensitivity';
        xLabel = '1-avg. Specificity';
        figureHandle = IO.getFiguresAndCombine(filesToCombine, summaryPath, figureName, xLabel, yLabel, legendNames);
    end
    
    function summarizeRankDependency(datasetName)
    % Produces a figure in our paper.
        import affiliationRecommendation.*;
        logPath = Constants.getLogPath(datasetName);
        summaryPath = Constants.getSummaryPath(datasetName);
        legendNames = {};
        
        filesToCombine = {};
        files = IO.listFiles([logPath 'aff*jointSVDKatzClustered-5cl-300f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzLFM-c(C, 5, 300)';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatzClustered-5cl-200f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzLFM-c(C, 5, 200)';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatzClustered-5cl-100f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzLFM-c(C, 5, 100)';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatz2-100f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzCS(100)';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatz2-200f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzCS(200)';
        
        files = IO.listFiles([logPath 'aff*jointSVDKatz2-300f*.mat']);
        filesToCombine{end+1} = files{end};
        legendNames{end+1} = 'tkatzCS(300)';
        
        figureName = 'summaryRankDependency';
        yLabel = 'avg. Sensitivity';
        xLabel = '1-avg. Specificity';
        figureHandle = IO.getFiguresAndCombine(filesToCombine, summaryPath, figureName, xLabel, yLabel, legendNames);
    end
    
    function summarizePerformanceAffiliationRanges(datasetName, percentiles)
    % Produces the latent factors predictors figure in our paper.
        import affiliationRecommendation.*;
        logPath = Constants.getLogPath(datasetName);
        summaryPath = Constants.getSummaryPath(datasetName);
        legendNames = {};
        
        filesToCombine = {};
        for i=2:length(percentiles)
            files = IO.listFiles([logPath 'aff*jointSVDSymm*Approx*-' num2str(percentiles(i)) '*.mat']);
            filesToCombine{end+1} = files{end};
            legendNames{end+1} = ['SVD(C) ' num2str(percentiles(i-1)) '-' num2str(percentiles(i))];
            files = IO.listFiles([logPath 'aff*KatzJoint3Approx*-' num2str(percentiles(i)) '*.mat']);
            filesToCombine{end+1} = files{end};
            legendNames{end+1} = ['tkatz(C) ' num2str(percentiles(i-1)) '-' num2str(percentiles(i))];
        end
        
        figureName = 'summaryAffiliationRanges';
        yLabel = 'avg. Sensitivity';
        xLabel = '1-avg. Specificity';
        figureHandle = IO.getFiguresAndCombine(filesToCombine, summaryPath, figureName, xLabel, yLabel, legendNames);
    end

    function testClass()
        display 'Class is ok!'
    end

end
end
