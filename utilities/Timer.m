classdef Timer
properties
    initTime;
    elapsedTime;
end

methods
    function timer = Timer
        timer.initTime = cputime;
%          fprintf(1,'Init time: %d\n', timer.initTime);
    end
    
    function timer = endTimer(timer)
        timer.elapsedTime = cputime - timer.initTime;
        fprintf(1,'  Time elapsed: %d\n', timer.elapsedTime);
    end
end

methods(Static=true)
    function timeStamp = getTimeStamp()
        timeStamp = datestr(datenum(clock()),'YYYY-mm-dd-HH:MM:SS:FFF');
    end
    
    function testClass
        display 'Class definition is ok';
    end
end
end