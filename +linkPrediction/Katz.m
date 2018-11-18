classdef Katz

methods(Static = true)

function [ScoreMatrix] = katz_exact(M, beta)
    % KATZ computes the Katz measure between two nodes in a social network.
    %
    % function [ScoreMatrix] = katz(M, beta)
    %
    % INPUT:
    %    M      (matrix) adjacency matrix of the graph, size n by n
    %    beta   (scalar) the damping parameter (default=0.05)
    %
    % OUTPUT:
    %    ScoreMatrix      (matrix) ScoreMatrix matrix, which is given by (I-beta*M)^(-1)-I
    
    if nargin < 2
        beta = 0.05;
    end
    
    I = speye(size(M));
    ScoreMatrix = (I-beta*M)\(beta*M);
    
end % kats function
    

function [ScoreMatrix] = KatzApprox(M, maxi, beta)
    % KATZ computes the Katz measure between two nodes in a social network approximately.
    %
    % function [ScoreMatrix] = KatzApprox(M, beta)
    %
    % INPUT:
    %    M      (matrix) adjacency matrix of the graph, size n by n
    %    idx1:  (matrix) 0/1 index matrix of nodes in the left side
    %    idx2:  (matrix) 0/1 index matrix of nodes in the right side
    %    maxi:  (scalar) maximum number of iterations
    %    beta   (scalar) the damping parameter (default=0.001)
    %
    % OUTPUT:
    %    ScoreMatrix      (matrix) ScoreMatrix matrix, which is given by sum_i^maxi beta^i*M^i 
    
    if nargin < 2
        maxi = 5;
    end
    if nargin < 3
        beta = 0.001;
    end
    
    tmp = beta*M;
    ScoreMatrix = full(tmp);
    for i = 2:1
        fprintf(1,'%d ',i)
        tmp = tmp*beta*M;
        ScoreMatrix = ScoreMatrix + tmp;
    end
    display(' ')
end % kats function


function res = Katz(A,At,e0,d,maxiter,dim,kmin) 
    %
    % [res] = KATZ(A,At,e0,d,maxiter,dim) compute a row or a column of the Katz matrix
    % 
    % The Katz matrix is obtained by
    %
    %       K = sum_k d^k * A^k
    %
    % res = K*e0 or K'*e0 (depending on whether dim = 2 or 1)
    %
    % Input:
    %
    %     A:       the adjacency matrix (needed only if dim = 1; otherwise can set to [])
    %     At:      At = A' (needed only if dim = 2; otherwise can set to [])
    %     e0:      an indicator vector/matrix. each column of e0 specifies
    %              a subset of columns/rows to aggregate
    %     d:       the dampening factor 
    %     maxiter: max number of iterations
    %     dim:     dim=1 => res = e0'*K
    %              dim=2 => res = K*e0
    %
    % Output:
    %
    % file:        Katz.m
    % directory:   /u/yzhang/SocialNet/src/
    % created:     Fri Sep 12 2008 
    % author:      Yin Zhang 
    % email:       yzhang@cs.utexas.edu
    %
    
    if nargin < 4, d = 0.05;     end
    if nargin < 5, maxiter = 10; end
    if nargin < 6, dim = 2;      end
    if nargin < 7, kmin = 1;     end
    
    n   = max(rows(A),rows(At));
    res = 0;
    e   = e0';
    
    if (dim == 2)
        for k = 1:maxiter
        % it turns out that At'*e is much faster than A*e
        e   = d*(e*At);
        if (k >= kmin)
            res = res + e;
        end
        end
        res = res';
    else % (dim == 1)
        for k = 1:maxiter
        % it turns out that A'*e is much faster than At*e
        e   = d*(e*A);
        if (k >= kmin)
            res = res + e;
        end
        end
    end
end

function KatzMatrix = getKatzMatrixFromPowers(powers, beta, iterations)
    KatzMatrix = beta*powers{1};
    if (nargin <= 2)
        iterations = length(powers);
    end
    for i = 2:iterations
        KatzMatrix = KatzMatrix + beta^i*powers{i};
    end
end


function Powers = getPowers(A, iterations)
    Powers = cell(iterations-1,1);
    Powers{1} = A;
    for i=2:iterations
        Powers{i} = Powers{i-1}*Powers{1};
    end
end

function [hybridPowers] = hybridPower(sources, maxDegree)
    % Compute the polynomial power terms given a set of matrices, normalized according to the frobenius norm.
    %
    % Input:
    %     sources: (1 by m cell array) each element is an n by n matrix
    %  maxDegree: max degree
    % Output:
    %    hybridPowers: (cell array of cell arrays) each element is a cell array with the nth cell array having n-degree products.
    
    % first compute power 2
    numSources = length(sources);
    hybridPowers = cell(maxDegree, 1);
    hybridPowers{1} = sources;
    
    for i = 2:maxDegree
        fprintf(1, '.');
        oldPowers = hybridPowers{i-1};
        newPowers = {};
        for j= 1:length(oldPowers)
            for k=1:numSources
                newPowers{end+1} = oldPowers{j}*sources{k};
                newPowers{end} = newPowers{end}/ norm(purePower, 'fro');
                newPowers{end+1} = sources{k}*oldPowers{j};
                newPowers{end} = newPowers{end}/ norm(purePower, 'fro');
            end
        end
        hybridPowers{i} = newSources;
    end
end % hybridPower function

end
end
