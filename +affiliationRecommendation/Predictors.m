classdef Predictors
methods(Static=true)
    function ScoreMatrix = randomPredictor(UserUser, Training, TargetEdgeSet, TrainingValid, ValidationEdgeSet, numPredictionsForValidation, initParams);
        ScoreMatrix = zeros(size(Training));
    end
end
end