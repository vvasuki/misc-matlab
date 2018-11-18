classdef Data
methods(Static = true)
function [UserUser, Training, TargetEdgeSet, TrainingValid, ValidationEdgeSet, clusters, remappingKeyForClustering] = getExperimentData(dataset, bGetNewSplits, fracEdgesToRemove, fracEdgesToRemoveValidation)
    clusters = [];
    remappingKeyForClustering = [];
    
    if(StringUtilities.isSubstring(dataset,'orkut') == 1 && StringUtilities.isSubstring(dataset,'Large') == 0)
        % dataFile is used to load the Social Network.
        dataFile = graph.Constants.ORK_SMALL_NET_9123_75546_gmin1_umin4;
        
        testNetworkFile = [graph.Constants.DATA_PATH, 'testNetworkOrkut.mat'];
        validationSetFile = [graph.Constants.DATA_PATH, 'trainingValidationSetOrkut.mat'];
        
        if(StringUtilities.isSubstring(dataset,'10cl') == 1)
            testNetworkClusters = [affiliationRecommendation.Constants.DATA_PATH_CLUSTERING 'orkut_10cl_l1.mat'];
        elseif(StringUtilities.isSubstring(dataset,'2cl') == 1)
            testNetworkClusters = [affiliationRecommendation.Constants.DATA_PATH_CLUSTERING 'orkut_2cl_l1.mat'];
        else
            testNetworkClusters = [affiliationRecommendation.Constants.DATA_PATH_CLUSTERING 'orkut_5cl_l1.mat'];
        end
    elseif(StringUtilities.isSubstring(dataset,'orkutLarge') == 1)
        % dataFile is used to load the Social Network.
        dataFile = graph.Constants.ORK_MATRECES_FILTERED_SOCIAL_NET;
        
        testNetworkFile = [graph.Constants.DATA_PATH, 'testNetworkOrkutLarge.mat'];
        validationSetFile = [graph.Constants.DATA_PATH, 'trainingValidationSetOrkutLarge.mat'];
        
        if(StringUtilities.isSubstring(dataset,'10cl') == 1)
            testNetworkClusters = [affiliationRecommendation.Constants.DATA_PATH_CLUSTERING 'orkutLarge_10cl_l1.mat'];
        elseif(StringUtilities.isSubstring(dataset,'2cl') == 1)
            testNetworkClusters = [affiliationRecommendation.Constants.DATA_PATH_CLUSTERING 'orkutLarge_2cl_l1.mat'];
        else
            testNetworkClusters = [affiliationRecommendation.Constants.DATA_PATH_CLUSTERING 'orkutLarge_5cl_l1.mat'];
        end
    elseif(StringUtilities.isSubstring(dataset,'youtube') == 1)
        % dataFile is used to load the Social Network.
        dataFile = graph.Constants.Y_SMALL_NET_16575_21326_gmin1_umin4;
        
        testNetworkFile = [graph.Constants.DATA_PATH, 'testNetworkYoutube.mat'];
        validationSetFile = [graph.Constants.DATA_PATH, 'trainingValidationSetYoutube.mat'];
        
        if(StringUtilities.isSubstring(dataset,'10cl') == 1)
            testNetworkClusters = [affiliationRecommendation.Constants.DATA_PATH_CLUSTERING 'youtube_10cl_l04.mat'];
        else
            testNetworkClusters = [affiliationRecommendation.Constants.DATA_PATH_CLUSTERING 'youtube_5cl_l04.mat'];
        end
    end
    
    if(strcmp(dataset,'artificial') == 1)
        numFactors = 60;
        numUsers = 500;
        numGroups = 600;
        fracNNZ = 0.12;
        socialNet = graph.SocialNet.getArtificialSocNet(numFactors, numUsers, numGroups, fracNNZ);
    else
        load(dataFile, 'socialNet');
    end
    
    UserUser = socialNet.userUser;
    if bGetNewSplits == 1
        if(StringUtilities.isSubstring(dataset,'Large') == 0)
            [Training, TargetEdgeSet] = linkPrediction.Predictor.removeRandomEdgesRowUniform(socialNet.userGroup, fracEdgesToRemove);
            display 'Done test vs training splitting!'
            
            [TrainingValid, ValidationEdgeSet] = linkPrediction.Predictor.removeRandomEdgesRowUniform(Training, fracEdgesToRemoveValidation);
            display 'Done validation splitting!'
            timeStamp = Timer.getTimeStamp();
            save(strrep(testNetworkFile, '.mat', [timeStamp '.mat']), 'Training', 'TargetEdgeSet', '-v7.3');  
        else
            socialNet.userGroup = sparse(socialNet.userGroup);
            [Training, TargetEdgeSet] = linkPrediction.Predictor.removeRandomEdges(socialNet.userGroup, fracEdgesToRemove);
            display 'Done test vs training splitting!'
        end
        save(strrep(validationSetFile, '.mat', [timeStamp '.mat']), 'TrainingValid', 'ValidationEdgeSet', '-v7.3');
        display 'Saved data to files!'
    else
        % Load existing ones.
        load(testNetworkFile);
        load(validationSetFile);
        if(StringUtilities.isSubstring(dataset,'Cluster'))
            [numUsers, numGroups] = size(Training);
            load(testNetworkClusters, 'lcc');
            connectedUsers = lcc(1:numUsers);
            connectedGroups = lcc(numUsers+1:end);
            UserUser = UserUser(connectedUsers, connectedUsers);
            Training = Training(connectedUsers, connectedGroups);
            TargetEdgeSet = TargetEdgeSet(connectedUsers, connectedGroups);
            TrainingValid = TrainingValid(connectedUsers, connectedGroups);
            ValidationEdgeSet = ValidationEdgeSet(connectedUsers, connectedGroups);
            
            load(testNetworkClusters, 'prtc', 'ind');
            clusters = prtc;
            remappingKeyForClustering = ind;
        end
    end
end


function testClass()
    display 'Class is ok!'
end
end
end
