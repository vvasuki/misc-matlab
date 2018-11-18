classdef StringUtilities
methods(Static=true)
    function [words] = split(string, delimiters)
        remainder = string;
        words = cell(0);
        while true
            if(exist('delimiters'))
                [str, remainder] = strtok(remainder, delimiters);
            else
                [str, remainder] = strtok(remainder);
            end
            if isempty(str),  break;  end
            numWords = length(words);
            words{numWords + 1} = str;
        end
    end
    
    function bIsSubstring = isSubstring(string, substring, bCaseSensitive)
        if(~exist('bCaseSensitive') || ~bCaseSensitive)
            bIsSubstring = ~isempty(findstr(lower(string), lower(substring)));
        else
            bIsSubstring = ~isempty(findstr(string, substring));
        end
    end
    
end
end