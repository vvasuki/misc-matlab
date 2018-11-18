classdef Constants
properties(Static = true)
    LOG_PATH = [system.SystemUtils.HOME_DIR '/projectsSvn/graphicalModels/log/'];
    TMP_PATH = [gmStructureLearningExperiments.Constants.LOG_PATH 'tmp/'];
end

methods(Static = true)
function testClass
    display 'Class definition is ok';
end
end
end