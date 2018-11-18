function katzJoint
fracEdgesToRemove = .3;
betaExp = -6;
numIterations = 3;
l = 0;

dataset = 'orkut';
if(strcmp(dataset,'orkut') == 1)
    dataFile = graph.Constants.ORK_SMALL_NET_10133_75551_gmin2;
    katz3AffFile = [graph.Constants.DATA_PATH, 'katz3Aff.mat'];
    validationSetFile = [graph.Constants.DATA_PATH, 'trainingValidationSet.mat'];
else
    dataFile = graph.Constants.Y_SMALL_NET_10228_7102_gmin2;
    katz3AffFile = [graph.Constants.DATA_PATH, 'katz3AffYoutube.mat'];
    validationSetFile = [graph.Constants.DATA_PATH, 'trainingValidationSetYoutube.mat'];
end

%  [socialNet, A, targetEdgeSet, numPredictions, precisionKatz, completenessKatz, numTargetEdges] = KatzAff.getKatzBaselineAff(fracEdgesToRemove, beta, numIterations);

%  save(katz3AffFile , 'A', 'targetEdgeSet', 'numPredictions', 'precisionKatz', 'completenessKatz', 'numTargetEdges', '-v7.3');

load(dataFile, 'socialNet');
load(katz3AffFile);

% Create validation and training matrices.
%  [Training, validationEdgeSet] = linkPrediction.Predictor.removeRandomEdges(A, fracEdgesToRemove);

%  save(validationSetFile , 'Training', 'validationEdgeSet', '-v7.3');
%  save(validationSetFileYoutube , 'Training', 'validationEdgeSet', '-v7.3');

% Or load these matrices from a file, instead of calculating them again.
load(validationSetFile);

display('Got validation edges. check it if you want.')
%  keyboard

numValidationSetEdges = full(sum(validationEdgeSet));
[numUsers, numGroups] = size(socialNet.userGroup);
socialNet.userUser = double(socialNet.userUser);


display 'Using validation to find the best l and betaExp'
lBest = 0.2;
betaExpBest = -1;
precisionBest = 0;
precisionOldBeta = 0;

%  Compute the powers
display 'Computing powers for training.';
tic
BA = socialNet.userUser*Training;
B2A = socialNet.userUser*socialNet.userUser*Training;
A3 = Training*Training'*Training;
toc

for betaExp=-12:1:0
precisionOldl = 0;
for l=0:0.2:2
    precision = getKatz(Training, BA, B2A, A3, validationEdgeSet, l, betaExp);
%      if precision<precisionOldl
%          break;
%      end
    
    if precision>precisionBest
        lBest = l;
        betaExpBest = betaExp;
        precisionBest = precision;
        fprintf(1,'Improved! betaExp, l: %d %d \n', betaExp, l);
    end
    
    precisionOldl = precision;
end %for l
    
%  if precisionOldl<precisionOldBeta
%  break;
%  end
precisionOldBeta = precisionOldl;
fprintf(1,'betaExp, precisionOldBeta: %d %d \n', betaExp, precisionOldBeta);
end %for beta

fprintf(1,'The final test: betaExpBest, lBest: %d %d \n', betaExpBest, lBest);
BA = socialNet.userUser*A;
B2A = socialNet.userUser*socialNet.userUser*A;
A3 = A*A'*A;

precision = getKatz(A, BA, B2A, A3, targetEdgeSet, lBest, betaExpBest)

display('All done, ready for your inspection.');
keyboard

end % katzJoint

function precision = getKatz(Training, BA, B2A, A3, validationEdgeSet, l, betaExp)
    fprintf(1,'betaExp, l: %d %d \n', betaExp, l);
    numPredictions = full(sum(validationEdgeSet));
    tic
    Score = l*BA + (10^betaExp)*(l*l*B2A + A3);
    [scoreVector, sortedIndices] = linkPrediction.Predictor.processSimilarityMatrix(Training, Score, numPredictions);
    [precision, completeness] = checkPrediction(scoreVector, sortedIndices, raining, predictedEdgeIndices, validationEdgeSet);
    toc
end
