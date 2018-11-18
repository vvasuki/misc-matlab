classdef SystemUtils
properties(Static=true)
    HOME_DIR = '/u/vvasuki/';
    LIBRARIES_PATH = '/public/linux/graft/';
    LIBRARIES_PATH_MATLAB = '/public/linux/graft/matlabToolboxes';
    JAVA_LIB_PATH = [system.SystemUtils.HOME_DIR 'vishvas/work/software/javaPackage/'];
end
methods(Static=true)

function enableMultiThreading()
    [returnValue, commandOutput] = system('cat /proc/cpuinfo|grep processor');
    numCPU = sum(commandOutput==':');
    % maxNumCompThreads('automatic')
    maxNumCompThreads(min(numCPU*2, 16));
    fprintf('Enabled %d threads for this %d processor machine.\n', maxNumCompThreads, numCPU);
end

function randomSeedFromClock()
    rand('twister',sum(100*clock))
    display('Initialized random seed with clock.');
end

function setJavaClasspath()
    import system.*;
    javaaddpath([SystemUtils.JAVA_LIB_PATH 'dist/javaPackage.jar']);
    javaaddpath([SystemUtils.JAVA_LIB_PATH '/lib/colt.jar']);
end

end
end