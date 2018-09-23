classdef BuildUtils
methods(Static=true)
function compileMexes(dir)
%  TODO: test.
    dir
    dirOld = pwd;
    cd(dir);
    mexFiles = what('*.c');
    mexFiles = mexFiles.mex;
    for mexFile=mexFiles
        mex(mexFile);
    end
    cd(dirOld);
end
end
end