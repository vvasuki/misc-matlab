classdef Parallelization
%  A bunch of methods related to IO operations on matreces.
methods(Static=true)

function machines = getTargetMachines()
    machines = {};
    targetMachinesUvanimor = 0:9;
    targetMachinesLhug = [0:7 9:11];
    
    
    for(i=targetMachinesUvanimor)
        machines{end+1} = ['uvanimor-' num2str(i)];
    end
    for(i=targetMachinesLhug)
        machines{end+1} = ['lhug-' num2str(i)];
    end
end

function jobStr = prepareMatlabJob(matlabCommand, fileNameStub, machineName)
    commandOutputFile = [fileNameStub '.out'];
    commandErrorFile = [fileNameStub '.err'];
    commandInputFile = [fileNameStub '.in'];
    matlabScriptCreationStr = sprintf('echo "%s; exit;">%s', matlabCommand, commandInputFile);
    jobStr = sprintf('%s; nohup ssh %s "matlab -nodisplay -nosplash <%s" >%s 2>%s &\n', matlabScriptCreationStr, machineName, commandInputFile, commandOutputFile, commandErrorFile);
end

function testClass
    display 'Class definition is ok';
end

end
end 
