classdef Constants
    properties(Static = true)
        DATA_PATH='/scratch/cluster/vvasuki/socialNW/';
        
        ORK_USER_USER_FILE_RAW=[graph.Constants.DATA_PATH, 'orkut-links.txt'];
        ORK_USER_GROUP_FILE_RAW=[graph.Constants.DATA_PATH, 'orkut-groupmemberships.txt'];
        ORK_USER_LIST=[graph.Constants.DATA_PATH, 'orkut-users.txt'];
        ORK_GROUP_LIST=[graph.Constants.DATA_PATH, 'orkut-groups.txt'];
        
        Y_USER_USER_FILE_RAW=[graph.Constants.DATA_PATH, 'youtube-links.txt'];
        Y_USER_GROUP_FILE_RAW=[graph.Constants.DATA_PATH, 'youtube-groupmemberships.txt'];
        Y_USER_LIST=[graph.Constants.DATA_PATH, 'youtube-users.txt'];
        Y_GROUP_LIST=[graph.Constants.DATA_PATH, 'youtube-groups.txt'];
        
        ORK_MATRECES_RAW=[graph.Constants.DATA_PATH, 'orkut_matrecesRaw.mat'];
        ORK_MATRECES_FILTERED=[graph.Constants.DATA_PATH, 'orkut_matrecesFiltered.mat'];
        ORK_MATRECES_FILTERED_SOCIAL_NET=[graph.Constants.DATA_PATH, 'orkut_matrecesFilteredSocialNet.mat'];
        ORK_SOCIAL_NET_5FRIENDED=[graph.Constants.DATA_PATH, 'ORK_SOCIAL_NET_5FRIENDED.mat'];
        
        Y_MATRECES_RAW=[graph.Constants.DATA_PATH, 'youtube_matricesRaw.mat'];
        Y_MATRECES_FILTERED_SOCIAL_NET=[graph.Constants.DATA_PATH, 'youtube_matricesFilteredSocialNet.mat'];
        
        ORK_SOCIAL_NET_2400856_12802 = [graph.Constants.DATA_PATH, 'ORK_SOCIAL_NET_2400856_12802.mat'];
        ORK_SMALL_NET_10133_75551_gmin2=[graph.Constants.DATA_PATH, 'ORK_SMALL_NET_10133_75551_gmin2.mat'];
        ORK_SMALL_NET_2047_6713_gmin2=[graph.Constants.DATA_PATH, 'ORK_SMALL_NET_2047_6713_gmin2.mat'];
        ORK_SMALL_NET_9123_75546_gmin1_umin4=[graph.Constants.DATA_PATH, 'ORK_SMALL_NET_9123_75546_gmin1_umin4.mat'];
        
        Y_SMALL_NET_10228_7102_gmin2=[graph.Constants.DATA_PATH, 'Y_SMALL_NET_10228_7102_gmin2.mat'];
        Y_SMALL_NET_16575_21326_gmin1_umin4=[graph.Constants.DATA_PATH, 'Y_SMALL_NET_16575_21326_gmin1_umin4.mat'];
        
    end
end