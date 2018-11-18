classdef Constants
properties(Static = true)
    LOG_PATH = '/u/vvasuki/projectsSvn/affiliationNwProject/log/';
    DATA_PATH = '/u/vvasuki/projectsSvn/affiliationNwProject/data/';
    DATA_PATH_CLUSTERING = [affiliationRecommendation.Constants.DATA_PATH 'clustering/'];
    Y_SMALL_NET_16575_21326_gmin1_umin4_1comp =[affiliationRecommendation.Constants.DATA_PATH_CLUSTERING, 'testNetworkYoutube'];
    TIMING_LOG_FILE = 'timingLog.txt';
end
methods(Static=true)
function summaryPath = getSummaryPath(datasetName)
    import affiliationRecommendation.*;
    summaryPath = [Constants.getLogPath(datasetName) 'summary/'];
end

function logPath = getLogPath(datasetName)
    import affiliationRecommendation.*;
    logPath = [Constants.LOG_PATH datasetName '/'];
end

function testClass
    display 'Class definition is ok';
end

end
end
