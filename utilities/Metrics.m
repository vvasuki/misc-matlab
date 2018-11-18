classdef Metrics
methods(Static=true)
    function kl = klDivergenceSymm(x, y)
    klxy = Metrics.klDivergence(x, y);
    klyx = Metrics.klDivergence(x, y);
    kl = (klxy + klyx)/2;
    end
    
    function klxy = klDivergence(x, y)
    logFactor = log(x./y);
    klxy = sum(x.*logFactor);
    end
end
end