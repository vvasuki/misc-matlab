classdef Graph
methods(Static = true)
function AdjMatrix = getGraph(topology, numNodes, degmax)
    import graph.*;
    %  Most of the following was copied from an implementation by pradeep ravikumar.
    if(isempty(degmax))
        degmax = (numNodes - 1);
    end
    
    AdjMatrix = logical(ones(numNodes, numNodes));
    AdjMatrix = AdjMatrix - diag(diag(AdjMatrix));
    
    switch topology
        case 'grid'
            %Grid Graph, 2D mesh
            gridSize = sqrt(numNodes);
            for i = 1:numNodes
                for j = (i+1):numNodes
                    if(Graph.noGridEdge(i,j,gridSize,gridSize))
                            AdjMatrix(i,j) = 0;
                            AdjMatrix(j,i) = 0;
                    end
                end
            end
        
        case '8ngrid'
            %Grid Graph, 2D mesh, 8 nn
            gridSize = sqrt(numNodes);
            for i = 1:numNodes
                for j = (i+1):numNodes
                    if(Graph.no8nGridEdge(i,j,gridSize,gridSize))
                            AdjMatrix(i,j) = 0;
                            AdjMatrix(j,i) = 0;
                    end
                end
            end
        
        case 'star'
            AdjMatrix = zeros(numNodes);
            sig = eye(numNodes);
            
            neighbors = 1 + randperm(numNodes-1);
            neighbors = neighbors(1:degmax);
            AdjMatrix(1,neighbors) = 1;
            AdjMatrix(neighbors,1) = 1;
            
        case 'tree1'
            %a specific tree or a random tree
            gridSize = sqrt(numNodes);
            AdjMatrix = zeros(numNodes,numNodes);
            for i = 1:gridSize
                    for j = 2:gridSize
                            cnode = gridSize * (i-1) + j;
                            pnode = cnode - 1;		
                            AdjMatrix(cnode,pnode) = 1;
                            AdjMatrix(pnode,cnode) = 1;
                    end
            end
            for i = 2:gridSize
                    pnode = gridSize * (i-1);
                    cnode = gridSize * i;
                    AdjMatrix(cnode,pnode) = 1;
                    AdjMatrix(pnode,cnode) = 1;
            end
        
        case 'chain'
            AdjMatrix = zeros(numNodes,numNodes);
            for i = 1:(numNodes-1)
                    AdjMatrix(i,i+1) = 1;
                    AdjMatrix(i+1,i) = 1;
            end

        case 'spade'
            AdjMatrix = Graph.getGraph('chain', numNodes, degmax);
            AdjMatrix(1, 3) = 1;
            AdjMatrix(3, 1) = 1;
            
        otherwise
            error('Not implemented');
    end %switch
    density = MatrixFunctions.getDensity(AdjMatrix);
    if(density <0.1)
        AdjMatrix = sparse(AdjMatrix);
    end
end

function SpanTree = getMaxSpanningTree(adj)
    import graph.*;
    SpanTree = Graph.getMinSpanningTree(-adj);
end

function tr = getMinSpanningTree(adj)
% Prim's minimal spanning tree algorithm
% Prim's alg idea:
%  start at any node, find closest neighbor and mark edges
%  for all remaining nodes, find closest to previous cluster, mark edge
%  continue until no nodes remain
% INPUTS: graph defined by adjacency matrix
% OUTPUTS: matrix specifying minimum spanning tree (subgraph)
% Gergana Bounova, March 14, 2005
    % check if graph is connected:
    import graph.*;
    if not(Graph.isConnected(adj))
      'graph is not connected, no spanning tree exists'
      return
    end

    n = length(adj); % number of nodes
    tr = zeros(n);

    adj(find(adj==0))=inf; % set all zeros in the matrix to inf

    conn_nodes = 1;        % nodes part of the min-span-tree
    rem_nodes = [2:n];     % remaining nodes

    while length(rem_nodes)>0
      [minlink]=min(min(adj(conn_nodes,rem_nodes)));
      ind=find(adj(conn_nodes,rem_nodes)==minlink);

      ind_j=ceil(ind(1)/length(conn_nodes));
      ind_i=mod(ind(1),length(conn_nodes));
      if ind_i==0
          ind_i=length(conn_nodes);
      end

      i=conn_nodes(ind_i); j=rem_nodes(ind_j); % gets back to adj indices
      tr(i,j)=1; tr(j,i)=1;
      conn_nodes = [conn_nodes j];
      rem_nodes = setdiff(rem_nodes,j);

    end

end

function nedge = no8nGridEdge(i,j,m,n)
    %  Most of the following was copied from an implementation by pradeep ravikumar.
    edge = false;
    if(mod(i,m) == 1)
        edge = (j == (i+1)) || (j == (i+m)) || (j == (i-m)) || (j == (i - m + 1)) || (j == (i +m + 1));
    else if(mod(i,m) == 0)
                edge = (j == (i-1)) || (j == (i+m)) || (j == (i-m)) || (j == (i - m - 1)) || (j == (i + m -1));
        else
                edge = (j == (i-1)) || (j == (i+1)) || (j == (i+m)) || (j == (i-m)) || (j == (i - m - 1)) || (j == (i + m -1)) || (j == (i - m + 1)) || (j == (i +m + 1));
        end
    end
    nedge = ~edge;
end

function nedge = noGridEdge(i,j,m,n)
    %  Most of the following was copied from an implementation by pradeep ravikumar.
    nedge = true;
    if(mod(i,m) == 1)
        nedge = ~((j == (i+1)) || (j == (i+m)) || (j == (i-m)));
    else if(mod(i,m) == 0)
            nedge = ~((j == (i-1)) || (j == (i+m)) || (j == (i-m)));
        else
            nedge = ~((j == (i+1)) || (j == (i-1)) || (j == (i+m)) || (j == (i-m)));
        end
    end
end

function bIsTree = isTree(topologyOrAdjMatrix)
    import graph.*;
    bIsTree = false;
    if(size(topologyOrAdjMatrix, 1)>1 && numel(topologyOrAdjMatrix)>1)
        adj = topologyOrAdjMatrix;
    else
        topology = topologyOrAdjMatrix;
    end
    if(exist('adj', 'var'))
        if(Graph.isConnected(adj) && Graph.getNumEdges(adj)==size(adj, 1)-1)
            bIsTree = true;
        end
    else
        if(StringUtilities.isSubstring(topology, 'star') || StringUtilities.isSubstring(topology, 'chain') || StringUtilities.isSubstring(topology, 'tree'))
            bIsTree = true;
        end
    end
end

% Determine if a graph is connected:
% works by finding a zero eigenvalue and if the corresponding eigenvector
% has 0s, then a sum of non-zero number of rows/columns of the adjacency is
% 0 hence the degrees of these nodes are 0 and the graph is disconnected
% INPUTS: adjacency matrix
% OUTPUTS: Boolean variable {0,1}
% Courtesy of Dr. Daniel Whitney, (idea by Ed Schneiderman) circa 2006
function S = isConnected(adj)
    adj = double(adj ~=0);
    n = length(adj);
    x = zeros(n,1);
    x(1)=1;

    while 1
         y = x;
         x = adj*x + x;
         x = x>0;
         if x==y
             break
         end
    end

    if sum(x)<n
         S = false;
    else
         S = true;
    end
end

function [extraEdgesInA, extraEdgesInB] = getDifference(AdjMatrixA, AdjMatrixB)
    import graph.*;
    % Input adjacency matrices are assumed to be symmetric.
    numEdgesInA = Graph.getNumEdges(AdjMatrixA);
    numEdgesInB = Graph.getNumEdges(AdjMatrixB);
    numCommonEdges = Graph.getNumEdges(AdjMatrixA & AdjMatrixB);
    extraEdgesInA = numEdgesInA - numCommonEdges;
    extraEdgesInB = numEdgesInB - numCommonEdges;
end

function numEdges = getNumEdges(AdjMatrix)
    if(size(AdjMatrix, 1) == size(AdjMatrix, 2))
        numEdges = full(sum(sum(triu(logical(AdjMatrix)) > 0)));
    else
        numEdges = full(sum(sum(logical(AdjMatrix) > 0)));
    end
end

% Finds the number of k-neighbors (k links away) for every node
% INPUTS: adjacency matrix, node index, k - number of links
% OUTPUTS: vector of k-neighbors indices
% Gergana Bounova, May 3, 2006
function kneigh = kNeighbors(adj,ind,k)
    adjk = adj;
    for i=1:k-1
      adjk = adjk*adj;
    end
    kneigh = find(adjk(ind,:)>0);
end

function maxDegree = getMaxDegree(AdjMatrix)
    maxDegree = max(sum(AdjMatrix, 1));
end

function A = getConnectedSupergraph(A)
    import graph.*;
    degrees = sum(A, 1);
    components = Graph.findConnectedComponents(A);
    for i=2:length(components)
        componentA = components{i-1};
        componentB = components{i};
        A(componentA(1), componentB(1)) = 1;
    end
    A = A | A';
end

function [deg,indeg,outdeg]=getDegrees(adj)

    indeg = sum(adj);
    outdeg = sum(adj');
    function S=isdirected(adj)
        S= (size(adj, 1) ~= size(adj, 2));
    end

    if isdirected(adj)
        deg = indeg + outdeg; % total degree
    else
        % undirected graph: indeg=outdeg
        deg = indeg + diag(adj)';  % add self-loops twice, if any
    end
end

% Algorithm for finding connected components in a graph
% Valid for undirected graphs only
% INPUTS: adj - adjacency matrix
% OUTPUTS: a list of the components comp{i}=[j1,j2,...jk}
% Gergana Bounova, Last updated: October 2, 2009
function comp_mat = findConnectedComponents(adj)
    import graph.*;
    [deg,indeg,outdeg]=Graph.getDegrees(adj);   % degrees
    comp_mat=cell(1);                  % initialize matrix


    function comp=find_conn_compH(adj,i)
        % heuristic for finding the conn component to which "i" belongs to
        % works well in practice for large datasets
        % INPUTS: adjacency matrix and index of the key node
        % OUTPUTS: all node indices of the nodes to which "i" belongs to
        import graph.*;

        neigh1=Graph.kNeighbors(adj,i,1);
        neigh1=unique([neigh1 i]); % add i to its own component

        while 1
            len0=length(neigh1);
            for j=1:len0
                neigh2=Graph.kNeighbors(adj,neigh1(j),1);
                neigh1=[neigh1, neigh2];
                neigh1=unique(neigh1);
            end
            if len0==length(neigh1)
                comp=neigh1;
                return
            end
        end
    end

    for i=1:length(deg)
        if deg(i)>0
            done=0;
            for x=1:length(comp_mat)
                if length(find(comp_mat{x}==i))>0   % i in comp_mat(x).mat
                    done=1;
                    break
                end
            end
            if not(done)
                comp=find_conn_compH(adj,i);
                comp_mat{length(comp_mat)+1}=comp;
            end

        elseif deg(i)==0
            comp_mat{length(comp_mat)+1}=[i];
        end
    end
    comp_mat=comp_mat(2:length(comp_mat));  % remove the comp_mat(1).mat=[]
end

function testClass
    display 'Class definition is ok';
end

end
end