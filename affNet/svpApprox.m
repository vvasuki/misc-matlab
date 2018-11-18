function svpApprox()
    numFactors = 60;
    maxIter = 5;
    fracEdgesToRemove = .5;
    uncertaintyFactor = 5;
    
    dataset = 'artificial';
    getNewSplits = 1;
    useBigMergedNet = 1;
    setGGUnknown = 1;
    getKnownEntriesFromKatz = 0; 
%      smoothenUserUser = 1;
    
    [socialNet, A, targetEdgeSet, Training, validationEdgeSet] = getExperimentData(dataset, getNewSplits, fracEdgesToRemove);

    [numUsers, numGroups] = size(socialNet.userGroup);
    
    % Get the KnownEntries matrix for the test missing value set.
    numPredictions = full(sum(targetEdgeSet));
    numUnknowns =  numPredictions * uncertaintyFactor; 
    
    if(getKnownEntriesFromKatz == 1)
        %  KnownEntries = getKnownEntriesAffCommonNbd(A, numUnknowns);
        % or load.
        KnownEntriesAffFile = [graph.Constants.DATA_PATH, 'MaskAff.mat'];
        %  save(KnownEntriesAffFile , 'KnownEntries', 'numUnknowns');
        load(KnownEntriesAffFile);
        error('Does not work. Need validation edge set.');
    else
        KnownEntries = getKnownEntriesSimulation(targetEdgeSet, A, uncertaintyFactor);
        KnownEntriesValidation = getKnownEntriesSimulation(validationEdgeSet, Training, uncertaintyFactor);
    end
    
    if(useBigMergedNet == 1)
        if(setGGUnknown == 1)
            GGKnownEntries = logical(zeros(numGroups));
        else
            GGKnownEntries = logical(ones(numGroups));
        end
        
        KnownEntries = [logical(ones(numUsers)) KnownEntries; KnownEntries' GGKnownEntries];
        KnownEntriesValidation = [logical(ones(numUsers)) KnownEntriesValidation; KnownEntriesValidation' GGKnownEntries];
    else
        KnownEntries = [KnownEntries logical(ones(numUsers))];
        KnownEntriesValidation = [KnownEntriesValidation logical(ones(numUsers))];
    end
    

    
    numTrainingEdges = full(sum(sum(A)));
    numTargetEdges = full(sum(targetEdgeSet));
    numPredictions =  numTargetEdges;
    
%      Prepare log file.
    file = fopen([graph.Constants.DATA_PATH, 'svpApprox'], 'a');
    
    fprintf(file, '\n\n****** %s ******\n', [graph.Constants.DATA_PATH, dataset]);
    fprintf(file, '%d\n', clock());
    fprintf(file, 'maxIter: %d, useBigMergedNet: %d getKnownEntriesFromKatz: %d setGGUnknown: %d fracEdgesToRemove: %d \n', maxIter, useBigMergedNet, getKnownEntriesFromKatz, setGGUnknown, fracEdgesToRemove);
    
    %  Set numFactors and l to be the best l and numFactors from validation.
    [numFactorsBest, lBest] = learnParameters(socialNet.userUser, Training, validationEdgeSet, useBigMergedNet, KnownEntriesValidation, maxIter, file);
    numFactors = numFactorsBest;
    l = lBest;
    
    precision = getSVPApprox(socialNet.userUser, A, l, KnownEntries, numFactors, targetEdgeSet, maxIter, useBigMergedNet);
    fprintf(file, 'on test set: numFactors: %d, l: %f, precisionSVP: %f\n', numFactors, l,  precision);
    fprintf(1, 'on test set: numFactors: %d, l: %f, precisionSVP: %f\n', numFactors, l,  precision);
    
    display('antyaM')
%      keyboard
    
end


function [numFactorsBest, lBest] = learnParameters(UserUser, UGTraining, validationEdgeSet, useBigMergedNet, KnownEntries, maxIter, file)
    display('Beginning validation!')
    numFactorsBest = 1;
    lBest = 1;
    precisionSVPBest = 0;
    l=0;
    for numFactors=40:10:100
        precisionSVPOld_l = 0;
        for l=0:0.2:3.0
            precisionSVP = getSVPApprox(UserUser, UGTraining, l, KnownEntries, numFactors, validationEdgeSet, maxIter, useBigMergedNet);
            fprintf(file, 'on validation set: numFactors: %d, l: %f, precisionSVP: %f\n', numFactors, l,  precisionSVP);
            fprintf(1, 'on validation set: numFactors: %d, l: %f, precisionSVP: %f\n', numFactors, l,  precisionSVP);
            if(precisionSVP < precisionSVPOld_l)
                display(' Searched l long enough!')
                break;
            end
            if(precisionSVPBest< precisionSVP)
                numFactorsBest = numFactors;
                lBest = l;
                precisionSVPBest = precisionSVP;
            end
            precisionSVPOld_l = precisionSVP;
        end
        if(precisionSVPOld_l < precisionSVPBest)
            display(' Searched numFactors long enough!')
            break;
        end
    end
    display('Done with validation!')
end

function precision = getSVPApprox(UserUser, UGTraining, l, KnownEntries, numFactors, targetEdgeSet, maxIter, useBigMergedNet)
    [numUsers, numGroups] = size(UGTraining);
    numPredictions = full(sum(targetEdgeSet));
    
    if(useBigMergedNet == 1)
        MergedNet = [l*UserUser UGTraining; UGTraining' sparse(numGroups, numGroups)];
    else
        MergedNet = [UGTraining l*UserUser]; 
    end
    
    UnknownIndices = find(KnownEntries ==0);
    OldMergedNet = MergedNet;
    display('getSVPApprox: Beginning iterations')
    for i=1:maxIter
%          Get low rank approximation of current MergedNet.
        fprintf(1, 'iteration %d \n', i);
        [UserFactors,S,GroupFactors] = svds(MergedNet, numFactors);
        Score = UserFactors*S*GroupFactors';
        
%          Set MergedNet to have known entries equal to the old Score matrix.
        if(i < maxIter)
            MergedNet = OldMergedNet;
            tic
            MergedNet(UnknownIndices) = 0;
            Tmp = zeros(size(MergedNet));
            % If there are very few unknown entries, the following is faster than copying known entries..
            Tmp(UnknownIndices) = Score(UnknownIndices);
            MergedNet = MergedNet + Tmp;
            toc
            
            OldMergedNet = MergedNet;
        end
    end
    
    display 'Done with the iterations.'
    if(useBigMergedNet == 1)
        Score = Score(1:numUsers, numUsers+1:end).* (~KnownEntries(1:numUsers, numUsers+1:end));
    else
        Score = Score(1:numUsers, 1:numGroups).* (~KnownEntries(1:numUsers, 1:numGroups));
    end
    display 'Identified the right scores.'
    
    [scoreVector, sortedIndices] = linkPrediction.Predictor.processSimilarityMatrix(UGTraining, Score, numPredictions);
    
    [precision, completeness] = checkPrediction(scoreVector, sortedIndices, GTraining, predictedEdgeIndices, targetEdgeSet);
end % End getSVDPrecisionMergedNet

function KnownEntries = getKnownEntriesSimulation(targetEdgeSet, Training, noiseFactor)
    % Mark all target edges unknown.
    display('getKnownEntriesSimulation...');
    [numUsers, numGroups] = size(Training);
    TargetMatrix = reshape(targetEdgeSet, numUsers, numGroups);
    KnownEntries = logical(zeros(numUsers, numGroups));
    KnownEntries = logical(KnownEntries + ~TargetMatrix);
    
    % Mark some other random entries as unknown.
    knownEntries = find(Training == 0);
    entriesToKnownEntries = randomizationUtils.Sample.sampleWithoutReplacement(knownEntries, noiseFactor*full(sum(targetEdgeSet)));
    KnownEntries(entriesToKnownEntries) = 0;
    KnownEntries = logical(KnownEntries);
    display('  done with getKnownEntriesSimulation...');
end