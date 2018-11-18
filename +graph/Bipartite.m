classdef Bipartite
%     The class of bipartite graphs ((A, B), E)
properties
    AB_AdjMatrix;
end
methods
    function graph=Bipartite(AB_AdjMat)
        graph.AB_AdjMatrix = AB_AdjMat;
    end
end

methods(Static = true)
    function adjMatrix = getAdjMatrix(bipartiteAdjMatrix)
        [rows, cols] = size(bipartiteAdjMatrix);
        adjMatrix = [zeros(rows, rows), bipartiteAdjMatrix;
                bipartiteAdjMatrix', zeros(cols, cols)];
    end
    
    function KnownEntries=get2HopLinks(Src, numUnknowns)
    %  Suppose we are interested in predicting links between row and column nodes which are two hops away (after adding edges among row nodes based on common column nodes). Then, this method returns a logical matrix which identifies such links.
    
    % We don't care to consider only the upper triangular portion of the score matrix as we are dealing with rectangular matrix.
        display('Calculating initial scores');
        InitScores = full(Src * Src' * Src);
        display('Setting scores corresponding to edges in Source to 0')
        tmpNzIndices = find(Src);
        display('.. Found necessary indices')
        tic
        InitScores (tmpNzIndices) = 0;
        toc
        
        display('getKnownEntriesAffCommonNbd: Sorting')
        tic
        [scoreVector indices] = sort(full(InitScores(:)),'descend');
        toc
        display('Sorted')
        unknownIndices = indices(1:numUnknowns);
        KnownEntries = logical(ones(size(InitScores)));
        KnownEntries(unknownIndices) = 0;
    end
end
end