classdef SocialNet
%     The class of bipartite graphs ((A, B), E)
    properties
        userGroup;
        userUser;

%          userGroup with rows normalized to sum to 1.
        userGroupRowNormalized;
        
        groups;
        groupClusters;
        userClusters;
        
        userMembershipCount;
        groupMembershipCount;
    end
methods
    function socialNet=SocialNet(userUserMat)
        socialNet.userUser = userUserMat;
    end
    
    function socialNet = setUserGroupMat(socialNet, userGroupMat, groups)
        socialNet.userGroup = userGroupMat;
        [numUsers, numGroups] = size(userGroupMat);
        if(nargin > 2)
            socialNet.groups = groups;
        else
            socialNet.groups = (1:numGroups)';
            
        end
    end
    
    function socialNet = updateMembershipCount(socialNet)
%      Update membership counts.
        socialNet.userMembershipCount =sum(socialNet.userGroup, 2);
        socialNet.groupMembershipCount=sum(socialNet.userGroup, 1);
%          size(socialNet.userMembershipCount)
%          size(socialNet.groupMembershipCount)
        disp('Updated membership count vectors.');
    end
    
    function calculateStatistics(socialNet)
% there are over $8*10^6$ groups and $2.78*10^6$ users! A certain community
% has $>3*10^{5}$ users (over $6*10^6$ actually).
        socialNet = updateMembershipCount(socialNet);
        
        display('Check sizes:')
        size(socialNet.groups)
        size(socialNet.userUser)
        size(socialNet.userGroup)
        
        display('Membership counts: users. mean, min, mode')
        mean(socialNet.userMembershipCount) 
        min(socialNet.userMembershipCount) 
% shows 117
        
        mode(socialNet.userMembershipCount) 
% shows 1.
        
        display('Friendship counts: users. mean, mode, low degree folk')
        userFriendshipCount = sum(socialNet.userUser);
        mean(userFriendshipCount)
        mode(userFriendshipCount)
        sum(userFriendshipCount <= 10)
%          In the matrix without low degree users being thrown away:
%          mean 79
%          mode 18
%          max friends, number of users
%          1, 13422
%          4, 76273
%          10, 252458
%          In the matrix with <5 degree users being thrown away:
%          mean 81.7
%          mode 16

        
        display('Membership counts: groups. mean, max, min, mode')
        mean(socialNet.groupMembershipCount) 
% shows 37.        
        max(socialNet.groupMembershipCount) 
        min(socialNet.groupMembershipCount) 
        
        mode(socialNet.groupMembershipCount) 
% of course 1.
% Considering the social network where people with <5 friends were not removed.
%          size, number of groups with that size:
%          100,  300160
%          150,  213189
%          200,  166605
%          300,  116960
%          500,  73700
%          1000, 38068
%          2000, 19393
%          3000, 12801
%          4000, 9504
%          5000, 7485
%          6000, 6090
%          7000, 5092
%          8000, 4455
%          9000, 3905
%          10000, 3498
        
%          sum(socialNet.groupMembershipCount== 1)
        sum(socialNet.groupMembershipCount > 10000)
    end
    
    function [rci, cci] = doITCC(socialNet)
%          Cocluster the user-group Matrix, and return the row-cluster indeces and the column cluster indices.
        userGroupMat = socialNet.userGroup;
        [numUsers, numGroups] = size(userGroupMat);
        numRowClusters = round(numUsers/1000);
        numColumnClusters = round(numGroups/30);
        [rci, cci, minobj, rec] = icc(userGroupMat, numRowClusters , numColumnClusters);
        
    end
    
    function [socialNet] = trawlAndGetSubNetwork(socialNet, maxUsers)
%          Input: maxUsers: the maximum number of users in the social network.
%          Output: socialNet derived from trawling the original user user graph.
        numUsers = size(socialNet.userUser,1);
%          Pick a random user
        user = randomizationUtils.Sample.sampleWithoutReplacement(1:numUsers,1);
        
%          Initialize the vectors.
        chosenUsers = sparse(numUsers,1);
        visitedUsers = sparse(numUsers,1);
        chosenUsers(user) = 1;
        
%          Grow the chosenUsers set.
        while(sum(chosenUsers - visitedUsers) >0 && sum(chosenUsers) < maxUsers)
%              Mark user as visited
            visitedUsers(user) = 1;
%              Get Children, add them to chosenUsers
            children = socialNet.userUser(:, user);
            chosenUsers = ((chosenUsers + children)>0);
            
            user = find(chosenUsers - visitedUsers);
            if(sum(chosenUsers - visitedUsers) >0)
                user = user(1);
            else
                display('All children visited! Space left: ')
                numUsers - sum(chosenUsers)
            end
        end
        
%          Take the set of chosen users, truncate the userUser, userGroup matrices.
        socialNet.userUser = socialNet.userUser(find(chosenUsers), find(chosenUsers));
        socialNet.userGroup = socialNet.userGroup(find(chosenUsers), :);
        
%          Discard groups without members
        socialNet = socialNet.removeTinyGroups(1);
    end
    
    function socialNet = pickLeadingGroups(socialNet, groupSize)
%          Want to run experiments on a small dataset. So, pick group with small size from the matreces. Return a SocialNet object with the top groupSize groups and connected users.
%          TODO: Write fn to pick the most informative groups instead.
%          Check sizes
        size(socialNet.groups)
        size(socialNet.userUser)
        size(socialNet.userGroup)
        
        socialNet = socialNet.updateMembershipCount();
%          Identify the leading groups.
        leadingGroups = find(socialNet.groupMembershipCount>groupSize);
        
%          Cut down the userGroup matrix.
        socialNet.userGroup = socialNet.userGroup(:, leadingGroups);
        
%          Cut down the groups vector.
        socialNet.groups = socialNet.groups(leadingGroups,1);
        
%          Check sizes
        size(socialNet.groups)
        size(socialNet.userUser)
        size(socialNet.userGroup)
        
%          Remove users without affiliations, thence cut down the userUser matrix.
        socialNet = socialNet.removeUsersFewAffiliations(0);

%          Check sizes
        size(socialNet.groups)
        size(socialNet.userUser)
        size(socialNet.userGroup)
        
        socialNet = socialNet.updateMembershipCount();
    end
    
    function socialNet = discardLargeGroups(socialNet, groupSize)
%          Check sizes
        size(socialNet.groups)
        size(socialNet.userUser)
        size(socialNet.userGroup)
        
        socialNet = socialNet.updateMembershipCount();
%          Identify the leading groups.
        smallGroups = find(socialNet.groupMembershipCount<=groupSize);
        
%          Cut down the userGroup matrix.
        socialNet.userGroup = socialNet.userGroup(:, smallGroups);
        
%          Cut down the groups vector.
        socialNet.groups = socialNet.groups(smallGroups,1);
        
%          Check sizes
        size(socialNet.groups)
        size(socialNet.userUser)
        size(socialNet.userGroup)
        
%  %          Remove users without affiliations, thence cut down the userUser matrix.
%          socialNet = removeUsersFewAffiliations(socialNet, 0);

%          Check sizes
        size(socialNet.groups)
        size(socialNet.userUser)
        size(socialNet.userGroup)
        
        socialNet = socialNet.updateMembershipCount();
    end
    
    function socialNet = removeUsersFewAffiliations(socialNet, minMembership)
%          Removes users without affiliations from the socialNet and returns it.
        socialNet = socialNet.updateMembershipCount();
        usersWithAffiliations = find(socialNet.userMembershipCount > minMembership);
        socialNet.userUser= socialNet.userUser(usersWithAffiliations, usersWithAffiliations);
        socialNet.userGroup=socialNet.userGroup(usersWithAffiliations, :);
        socialNet = socialNet.updateMembershipCount();
    end
    
    function socialNet = removeTinyGroups(socialNet, minGroupSize)
%          Removes groups with fewer than minGroupSize users and returns the modified socialNet.
        socialNet = socialNet.updateMembershipCount();
        groupsWithAffiliations = find(socialNet.groupMembershipCount > minGroupSize-1);
%          size(groupsWithAffiliations )
        socialNet.userGroup=socialNet.userGroup(:,groupsWithAffiliations);
        socialNet.groups = socialNet.groups(groupsWithAffiliations, :);
        socialNet = socialNet.updateMembershipCount();
    end

    function socialNet = pickRandomUsers(socialNet, numUsersFinal)
%         picks a random set of users, adjusts userUser and userGroup matrices, returns the modified socialNet.
        numUsersInit = size(socialNet.userUser, 1);
        sampledUsers = randomizationUtils.Sample.sampleWithoutReplacement(1:numUsersInit, numUsersFinal);
        socialNet.userUser = socialNet.userUser(sampledUsers,sampledUsers);
        socialNet.userGroup = socialNet.userGroup(sampledUsers, :);
        socialNet = socialNet.removeTinyGroups(1);
        socialNet = socialNet.updateMembershipCount();
    end

    function doKernelKmeans(socialNet)
%          Incomplete.
        userUserKatz = linkPrediction.Katz.KatzApprox(socialNet.userUser, 5, .05);
        
    end
    
    function checkClusterStatistics(socialNet)
        [numGroups, numClusters] = size(socialNet.groupClusters)
        clusterSizes = sum(socialNet.groupClusters, 1);
        max(clusterSizes)
        min(clusterSizes)
        mean(clusterSizes)
        
        cluster = 23;
        memberGroupIDs = socialNet.getGroupClusterMemberIDs(cluster)
    end
    
    function memberGroupIDs = getGroupClusterMemberIDs(socialNet, cluster)
        [numGroups, numClusters] = size(socialNet.groupClusters);
        if(cluster> numClusters)
            memberGroupIDs = [];
            return
        end
        clusterMembers = find(socialNet.groupClusters(:,cluster));
        memberGroupIDs = socialNet.groups(clusterMembers, 1);
    end
    
    function socialNet = cocluster(socialNet)
%          cluster the groups of the socialNet, record this in the groupClusters and the userClusters matrices. Return the modified social network.


%          Eliminate groups for which are too tiny to be clustered reliably based on user membership.
        socialNet = socialNet.removeTinyGroups(10);
        
%          Prepare userGroupRowNormalized.

        socialNet = socialNet.updateMembershipCount();
        rowNormalizer = diag(socialNet.userMembershipCount);
        socialNet.userGroupRowNormalized = inv(rowNormalizer)*socialNet.userGroup;
        
%          Gather some required values.
        numGroups = size(socialNet.groups, 1);

%          Initialize groupClusters.
        socialNet.groupClusters = sparse(numGroups,0);

%          Identify strong clusters.
        for group = 1:numGroups
            
            closestGroups = socialNet.identifyClosestGroups(group);
            closestGroups(end + 1) = group;
            socialNet = socialNet.clusterGroupsTogether(closestGroups);
        end
%          TODO: Now, try to assign clusters to previously unclustered groups, by finding strength of association of each (group, cluster) pair.
        
%          TODO: remove 0 or 1 sized clusters from the groupClusters matrix.
        clusterSizes = sum(socialNet.groupClusters, 1);
        nonEmptyClusters  = find(clusterSizes > 1);
        socialNet.groupClusters = socialNet.groupClusters(:, nonEmptyClusters);
    end
    
    function closestGroups = identifyClosestGroups(socialNet, group)
%          Identify groups closest to group in socialNet.
        members = find(socialNet.userGroup(:, group));
        memberAffiliations = sum(socialNet.userGroupRowNormalized(members, :),1);
        memberAffiliations(1, group) = 0;
        maxMemberAffiliation = max(memberAffiliations);
        if(maxMemberAffiliation < socialNet.userGroupRowNormalized(members, group)*.5)
            closestGroups = [];
        else
            closestGroups = find(memberAffiliations == maxMemberAffiliation);
        end
    end

    function socialNet = clusterGroupsTogether(socialNet, similarGroups)
%          Input: similarGroups: groups which need to be clustered together.
%          Output: the modified SocialNet object, in which groups listed in the similarGroups, and groups these groups are clustered with, are clustered together. This new clustering is indicated in the groupClusters matrix.
%          Tested and OK.
%          Handle degenerate cases, when the number of similar groups to be clustered is < 2:
        if(numel(similarGroups) < 2)
            socialNet = socialNet;
            return;
        end
%          Get some necessary values
        groupClusters = socialNet.groupClusters;
        numClusters = size(groupClusters,2);
        numGroups = size(socialNet.groups,1);
        
%          If groupClusters has 0 columns, add a column.
        if(numClusters == 0)
            groupClusters = sparse(numGroups, 1);
        end
        
%          Identify clusters which encompass groups in similarGroups. Thence get clustersToMerge vector.
        clusterMemberships = sum(groupClusters(similarGroups, :),1);
        clustersToMerge = find(clusterMemberships > 0);
        
%          Depending on the clustersToMerge:
%          Identify the newCluster number to encompass groups to be clustered to-gether.
%          Identify groups which have previously been clustered with groups in similarGroups: thence get relatedGroups, and add them to the new cluster.
        if(numel(clustersToMerge)==0)
            newCluster = numClusters + 1;
        else
            newCluster = clustersToMerge(1);
            relatedClustersMembership = sum(groupClusters(:,clustersToMerge),2);
            relatedClustersMembership = relatedClustersMembership > 0;
            
            groupClusters(:, newCluster) = relatedClustersMembership;
        end
%          Add groups in the similarGroups vector to the new cluster.
%          groupClusters
        groupClusters(similarGroups, newCluster) = ones(size(similarGroups));
%          groupClusters

%          If clustersTomerge > 1, identify the redundant clusters to empty, and empty them.
        if(numel(clustersToMerge)>1)
            clustersToEmpty = clustersToMerge(2:numel(clustersToMerge));
%              clustersToEmpty
            groupClusters(:, clustersToEmpty) = zeros(numGroups, numel(clustersToEmpty));
        end
%          groupClusters
        socialNet.groupClusters = groupClusters;
    end
    
    function socialAffNw = mergeSocialAffiliationNetworks(socialNet)
%          Take socialNet.userUser and socialNet.userGroup and merge these graphs to form a single graph, return this graph.
        [users, groups] = size(socialNet.userGroup);
        userUserEdgeWeight = 5;
        socialAffNw = [userUserEdgeWeight*socialNet.userUser, socialNet.userGroup;
        socialNet.userGroup', sparse(groups, groups)];
    end
    
    function [partition, obj] = mergeAndDoGraclus(socialNet)
        socialAffNw = socialNet.mergeSocialAffiliationNetworks();
        [partition, obj] = graclus(socialAffNw, 70);
        groupsPartition = partition(10001:end);
        clusterNumbers = unique(groupsPartition);
        clusteredGroupIndices = find(groupsPartition == 1);
        socialNet.groups(clusteredGroupIndices)
%          Run details:
%              With graph.Constants.ORK_SMALL_NET_10000_3498, giving equal weight to both sorts of edges, 
%              yields somewhat coherent, somewhat confusing groups.
%              cluster 1 seemed to be somehow related to western lifestyle: wine, gay sex, music, washington DC, IRC.
%              cluster 2: SUNY, pets, JSPNet company, iced tea.
%              cluster 3: iit kanpur, california highschool alumni, seattle something, google djs, certain umich alumni, western washington uni alumni.
%              Giving wt 5 to userUser edges does seem to alter cluster 1, but not by much.
    end

    function [partition, obj] = doGraclusUserUser(socialNet, numClusters)
        [partition, obj] = graclus(socialNet.userUser, numClusters);
    end
    
    function socialNet = removeUsersWithFewFriends(socialNet, minFriends)
        userFriendCount = sum(socialNet.userUser, 2);
        usersWithFriends=find(userFriendCount >= minFriends);
        size(usersWithFriends)
%       2781775 such folks
        
        socialNet.userUser= socialNet.userUser(usersWithFriends, usersWithFriends);
        socialNet.userGroup=socialNet.userGroup(usersWithFriends, :);
        
%          Check sizes
        size(socialNet.userUser)
        size(socialNet.userGroup)
        
        socialNet = socialNet.removeTinyGroups(1);

%          Check sizes
        size(socialNet.groups)
        size(socialNet.userUser)
        size(socialNet.userGroup)
        
    end

    function socialNet = getSmallNetwork(socialNet, minGroupSize, numUsers, minFriends)
        socialNet = socialNet.pickLeadingGroups(minGroupSize);
        socialNet.calculateStatistics();
        socialNet = socialNet.pickRandomUsers(numUsers);
        socialNet.calculateStatistics();
        socialNet = socialNet.removeUsersWithFewFriends(minFriends);
        socialNet.calculateStatistics();
    end

end

methods(Static = true)
    
    function socialNet = getArtificialSocNet(numFactors, numUsers, numGroups, fracNNZ)
        % Create artificial dataset to check how good SVD is at recovering missing links.
        
        UserFactors = rand(numUsers, numFactors);
        GroupFactors = rand(numGroups, numFactors);
        
        display ('Creating UserUser and UserGroup matrices.')
        UserUser = UserFactors*UserFactors';
        UserGroup = UserFactors*GroupFactors';
        UserUser = UserUser - diag(diag(UserUser));
        
        numUserUserLinks = floor(fracNNZ * numUsers * (numUsers - 1) / 2);
        numUserGroupLinks = floor(fracNNZ * numGroups * numUsers);
        UserUser = triu(UserUser);
        
        display('Sorting scores...')
        [userUserAffinity, indices] = sort(UserUser(:));
        UserUser = zeros(numUsers);
        UserUser(indices(1:numUserUserLinks)) = 1;
        
        [userGroupAffinity, indices] = sort(UserGroup(:));
        UserGroup = zeros(numUsers, numGroups);
        UserGroup(indices(1:numUserGroupLinks)) = 1;
        
        socialNet = graph.SocialNet(UserUser);
        socialNet.userGroup = UserGroup;
        
%          display('All done. Time to inspect.')
%          keyboard
    end 
    
    function socialNet = makeOrkutSocialNet()
%         Read the orkut user vs user and user vs group matreces stored in
%         files, make a SocialNet object thence, return this object.
%          tmp = load(graph.Constants.ORK_SMALL_NET_10000_3498);
%          socialNet = tmp.socialNet;
        tmp = load(graph.Constants.ORK_MATRECES_FILTERED);
        userGroupMat = tmp.userGroupMat;
        userUserMat = tmp.userUserMat;
        socialNet=graph.SocialNet(userUserMat);
        socialNet.userGroup = userGroupMat;
        socialNet.groups = tmp.groups;
    end
    
    function makeRawMatrix(userListFile, userUserFile, userGroupFile, numGroups, matrixFile)
%         Create userUser and userGroup matrices from raw orkut data and
%         write it to files.
        
%          graph.SocialNet.makeRawMatrix(graph.Constants.Y_USER_LIST, graph.Constants.Y_USER_USER_FILE_RAW, graph.Constants.Y_USER_GROUP_FILE_RAW, 30087, graph.Constants.Y_MATRECES_RAW);
        
        
%          Groups are 1:8730859 for orkut, 30087 for youtube.
        groups=1:numGroups;

%          read in users list
        users=load(userListFile);
        numUsers = size(users,1);
%          Returns 3072448
        numUsers = max(users);
%          3072632
        
        userUserLinks=load(userUserFile);
%          Check number of users with links to other users.
%          size(unique(userUserLinks))
%          Returns: 3072441
        
        userGroupLinks=load(userGroupFile);

%          Check if number of groups with affiliations is 8730859.
%          size(unique(userGroupLinks(:,2)))
%          Returns 8730857

%          Check number of users with affiliations.
%          size(unique(userGroupLinks(:,1)))
%          Returns 2783196
        
        %  Remove users without affiliations from both userUserMat and userGroupMat.
        userUserMat = MatrixTransformer.getPatternMatrix(userUserLinks, numUsers, numUsers);
        userUserMat = userUserMat(users,users);
%          It contains 3072448-3072441 rows and columns for users without friends.
        userGroupMat = MatrixTransformer.getPatternMatrix(userGroupLinks, numUsers, numGroups);
        userGroupMat = userGroupMat(users,:);
%          This yields a 3072448 * 8730859 matrix.

%          Check sizes
        size(users)
        size(userUserMat)
        size(userGroupMat)
        
        save(matrixFile, 'userUserMat', 'userGroupMat', 'users', '-v7.3');
    end
        
    function makeFilteredMatrices(matFile, numGroups, filteredMatFile)
%      Takes the matrices generated by makeRawMatrix(), symmetrifies the user-user matrix, removes users without affiliations, removes users with no friends.

%  graph.SocialNet.makeFilteredMatrices(graph.Constants.Y_MATRECES_RAW, 30087, graph.Constants.Y_MATRECES_FILTERED_SOCIAL_NET)

        load(matFile);
%          numGroups = 8730859 for orkut;
        groups = 1:numGroups;
        
%          Symmetrify userUserMat
        userUserMat = triu(userUserMat)' + tril(userUserMat)' + triu(userUserMat) + tril(userUserMat);
        userUserMat = (userUserMat>=1);

%          Check symmetry
        display 'Checking symmetry'
        sum(sum(triu(userUserMat)))
        sum(sum(tril(userUserMat)))
        display 'Checking symmetry: done'
        
%          Check number of groups with non zero memberships.
        groupMembershipCount=sum(userGroupMat, 1);
        groupsWithAffiliations = find(groupMembershipCount > 0);
        size(groupsWithAffiliations )
%          There are 2 groups with zero memberships.
        
        userMembershipCount=sum(userGroupMat, 2);
        usersWithAffiliations = find(userMembershipCount > 0);
%        2783196  such users, as expected.
        users= users(usersWithAffiliations, :);
        userUserMat= userUserMat(usersWithAffiliations, usersWithAffiliations);
        userGroupMat=userGroupMat(usersWithAffiliations, :);
        
%          Check sizes
        size(users)
        size(userUserMat)
        size(userGroupMat)
        
%          Create a SocialNet object
        socialNet = graph.SocialNet(userUserMat);
        socialNet.userGroup = userGroupMat;
        socialNet.groups = groups';
        
        socialNet = socialNet.removeUsersWithFewFriends(1);
%          Remove users with few friends.
        
        save(filteredMatFile, 'socialNet', '-v7.3');
    end
    

end
end

