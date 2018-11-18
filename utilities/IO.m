classdef IO
%  A bunch of methods related to IO operations on matreces.
methods(Static=true)
function toccs(mat, name)
    % toccs(mat, name) 
    %   convert a sparse matrix in Matlab to CCS binary format.

    % Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
    % $Id: toccs.m,v 1.3 2008/05/17 02:10:38 wtang Exp $

    [r,c] = size(mat);
    [i,j,xx] = find(mat);
    j = cumsum(histc(j, 0:c));

    fid = fopen(strcat(name, '.bin'), 'wb');
    fwrite(fid, [r c length(xx)], 'uint32');
    fwrite(fid, j, 'uint32');
    fwrite(fid, i-1, 'uint32');
    fwrite(fid, xx, 'double');
    fclose(fid);
end

function [mat] = fromccs(name)
    % [mat] = fromccs(name) 
    %   convert a sparse matrix in CCS format to Matlab.

    % Copyright 2008 Wei Tang (wtang@cs.utexas.edu)
    % $Id: fromccs.m,v 1.1 2008/05/17 02:10:38 wtang Exp $

    fid = fopen(strcat(name, '.bin'), 'rb');
    r = fread(fid, 1, 'uint32');
    c = fread(fid, 1, 'uint32');
    nz = fread(fid, 1, 'uint32');
    colptr = double(fread(fid, c+1, 'uint32'));
    rowidx = double(fread(fid, nz, 'uint32'));
    xx = fread(fid, nz, 'double');
    fclose(fid);
    mat = getSparse(r, c, rowidx, colptr, xx);
end

function clearClasses()
    clear classes; rehash pathreset;
end

function [figureHandle, fullFileNameSansExtension] = plotAndSave(xData, yData, xLabel, yLabel, filePrefix, figureName, figureTitle, legendNames, figureHandle, legendLocation)
    % Create multiple plots in a figure. 
    %      After legendNames, all arguments are optional.
    % E.g. xData = {[1:10], [1:10], [1:10], [1:10], [1:10], [1:10]}; 
    %      yData = yData = {[1:10], [2:2:20], [3:3:30], [4:4:40], [5:5:50], [6:6:60]};
    %      legendNames = {'y=x', 'y=2x', 'y=3x', 'y=4x', 'y=5x', 'y=6x'};
    %      IO.plotAndSave(xData, yData, 'test_x', 'test_y', 'path/to/file', 'testFile', legendNames, 'Test');
    if(nargin<9)
        figureHandle = figure;
    end

    bShowLengend = true;
    if(iscell(yData) == false)
        yData = {yData};
        legendNames = {figureName};
        bShowLengend = false;
    end
    if(iscell(xData) == false)
        xData = {xData};
    end
    
    if(isempty(legendNames))
        legendNames = {figureName};
        bShowLengend = false;
    end
    numPlots = length(xData);
    if(nargin<7)
        figureTitle = '';
    end
    
    if(nargin<10)
        legendLocation = 'Best';
    end
    
    markers = '+o*xsdph^v';
    % Note: . is omitted from the above.
    lineStyles = {'-', '--', '-.', ':'};
    colors = 'rbmk';
    % Note: ycg is omitted above.
    
    % Create plot
    currStyle = 0;
    currMarker = 0;
    currColor = 0;
    fontSize = 22;
    legendFontSize = 20;
    lineWidth = 2;
    markerSize = 6;
    
    for plotNum = 1:length(yData)
        currMarker = mod(currMarker, length(markers)) + 1;
        currStyle = currStyle + double(currMarker == 1);
        currColor = mod(currColor,length(colors)) + 1;

        x = xData{min(plotNum, length(xData))};
        y = yData{plotNum};

        plot(x, y, [lineStyles{currStyle} markers(currMarker) colors(currColor)], 'DisplayName', legendNames{plotNum}, 'LineWidth', lineWidth, 'MarkerSize', markerSize);
        set(gca, 'FontSize', fontSize);
        hold on
    end
    % Create xlabel
    xlabel({xLabel});
    % Create ylabel
    ylabel({yLabel});
    % Set title to the plot    
    figureTitle
    title(figureTitle);
    
    % Set the legend.
    if(bShowLengend)
        h_legend = legend(legendNames, 'Location', legendLocation);
        set(h_legend,'FontSize', legendFontSize);
    end
    grid on;
    hold off
    fullFileNameSansExtension = IO.saveFigure(figureHandle, filePrefix, figureName, xData, yData, legendNames);
    save([fullFileNameSansExtension '.mat'],'xLabel', 'yLabel', 'figureTitle', '-append');
end

function name = getHostname()
    [ret, name] = system('hostname');
end

function [figureHandle, fullFileNameSansExtension] = getFiguresAndCombine(files, filePrefix, figureName, xLabel, yLabel, legendNames)
    numFiles = length(files);
    xData = {};
    yData = {};
    figureTitle = '';
    
    bGetLegendNamesFromFigures = false;
    if(~exist('legendNames') || isempty(legendNames))
        bGetLegendNamesFromFigures = true;
        legendNames = {};
    end
    
    for i=1:numFiles
        file = files{i};
        data = load(file);
        if(iscell(data.xData))
            numPlots = length(data.xData);
            xData{end+1: end+numPlots} = data.xData{1:end};
            yData{end+1: end+numPlots} = data.yData{1:end};
        else
            numPlots = 1;
            xData{end+1: end+numPlots} = data.xData;
            yData{end+1: end+numPlots} = data.yData;
        end
        if(bGetLegendNamesFromFigures)
            legendNames{end+1: end+numPlots} = data.legendNames{1:end};
        end
    end

    if(~exist('xLabel', 'var') || isempty(xLabel))
        xLabel = data.xLabel;
        yLabel = data.yLabel;
    end
    [figureHandle, fullFileNameSansExtension] = IO.plotAndSave(xData, yData, xLabel, yLabel, filePrefix, figureName, figureTitle, legendNames);
end

function [figureHandle, fullFileNameSansExtension] = concatenatePlots(files, filePrefix, figureName, xLabel, yLabel)
    numFiles = length(files);
    xData = {};
    yData = {};
    figureTitle = '';
    function [data] = getPlotData(fileNum)
        file = files{fileNum};
        data = load(file);
    end

    data = getPlotData(1);
    xData = data.xData;
    yData = data.yData;
    legendNames=data.legendNames;
    numPlots=length(xData);
    
    for i=2:numFiles
        data = getPlotData(i);
        for j=1:numPlots
            xData{j}=[xData{j}(:); data.xData{j}(:)];
            yData{j}=[yData{j}(:); data.yData{j}(:)];
            [xData{j} rearrangedIndices] = sort(xData{j});
            yData{j} = yData{j}(rearrangedIndices);
        end    
    end
    [figureHandle, fullFileNameSansExtension] = IO.plotAndSave(xData, yData, xLabel, yLabel, filePrefix, figureName, figureTitle, legendNames);
end

function [figureHandle] = addDataPoints(fullFileNameSansExtension, xDataToAdd, yDataToAdd, xLabel, yLabel)
    xData = {};
    yData = {};
    figureTitle = '';
    data = load([fullFileNameSansExtension '.mat']);
    xData = data.xData;
    yData = data.yData;
    if(~exist('xLabel', 'var'))
        xLabel = data.xLabel;
        yLabel = data.yLabel;
        filePrefix = data.filePrefix;
    end
    legendNames=data.legendNames;
    numPlots=length(xData);
    for(j=1:numPlots)
        xData{j}=[xData{j}(:); xDataToAdd{j}(:)];
        yData{j}=[yData{j}(:); yDataToAdd{j}(:)];
        [xData{j} rearrangedIndices] = sort(xData{j});
        yData{j} = yData{j}(rearrangedIndices);
    end
    
    [figureHandle, fullFileNameSansExtension] = IO.plotAndSave(xData, yData, xLabel, yLabel, fullFileNameSansExtension, '', figureTitle, legendNames);
end

function [figureHandle] = removeDataPoint(fullFileNameSansExtension, xDataToRemove, xLabel, yLabel)
    xData = {};
    yData = {};
    figureTitle = '';
    data = load([fullFileNameSansExtension '.mat']);
    xData = data.xData;
    yData = data.yData;
    legendNames=data.legendNames;
    numPlots=length(xData);
    for(j=1:numPlots)
        for(ptToRemove = xDataToRemove)
            indicesToRemove = find(xData{j} == ptToRemove);
            xData{j}(indicesToRemove) = [];
            yData{j}(indicesToRemove) = [];
        end
    end

    [figureHandle, fullFileNameSansExtension] = IO.plotAndSave(xData, yData, xLabel, yLabel, fullFileNameSansExtension, '', figureTitle, legendNames);
end

function [figureHandle] = removeLegendsAndTitle(fullFileNameSansExtension, xLabel, yLabel)
    data = load([fullFileNameSansExtension '.mat']);
    if(~exist('xLabel', 'var'))
        xLabel = data.xLabel;
        yLabel = data.yLabel;
    end

    [figureHandle, fullFileNameSansExtension] = IO.plotAndSave(data.xData{1}, data.yData{1}, xLabel, yLabel, fullFileNameSansExtension, '', '');
end

function [figureHandle] = replaceAxis(fullFileNameSansExtension, axisId, newLabel, vectorFn, otherLabel)
    data = load([fullFileNameSansExtension '.mat']);
    xData = data.xData;
    yData = data.yData;
    if(isfield(data, 'xLabel'))
        xLabel = data.xLabel;
        yLabel = data.yLabel;
        legendNames=data.legendNames;
        figureTitle = data.figureTitle;
    else
        legendNames={};
        figureTitle = {};
    end
    numPlots=length(xData);
    for(j=1:numPlots)
        if(axisId == 1)
            if(exist('vectorFn'))
                xData{j} = vectorFn(xData{j});
            end
            xLabel = newLabel;
            if(exist('otherLabel'))
                yLabel = otherLabel;
            end
        else
            if(exist('vectorFn'))
                yData{j} = vectorFn(yData{j});
            end
            yLabel = newLabel;
            if(exist('otherLabel'))
                xLabel = otherLabel;
            end
        end
    end

    [figureHandle, fullFileNameSansExtension] = IO.plotAndSave(xData, yData, xLabel, yLabel, fullFileNameSansExtension, '', figureTitle, legendNames);
end

function fullFileNameSansExtension = saveFigure(figureHandle, filePrefix, figureName, xData, yData, legendNames)
    % Save the results
    timeStamp = Timer.getTimeStamp();
    if(isempty(figureName))
        fullFileNameSansExtension = filePrefix;
    else
        fullFileNameSansExtension = [filePrefix figureName timeStamp];
    end
    saveas(figureHandle, [fullFileNameSansExtension '.fig'], 'fig');
    saveas(figureHandle, [fullFileNameSansExtension '.jpg'], 'jpg');
    saveas(figureHandle, [fullFileNameSansExtension '.eps'], 'psc2');
    if(nargin>3)
        save([fullFileNameSansExtension '.mat']);
    end
end

function files = listFiles(searchPattern)
    filesString = ls(searchPattern);
    files = StringUtilities.split(filesString);
end

function selectivelyGatherFiles(files, marshallingArea, filePattern)
    if(exist(marshallingArea))
        mkdir(marshallingArea);
    end
    marshallingArea = [marshallingArea '/'];
    numFiles = length(files);
    for(i = 1:numFiles)
        file = files{i};
        bPackageFile = ~isempty(strfind(file, filePattern));
        if(~bPackageFile), continue; end;
        system(['cp -r ' file '/+* ' marshallingArea]);
        system(['cp -r ' file '/*.m ' marshallingArea]);
    end
    system(['rm -rf `find ' marshallingArea ' -type d -name .svn`']);
end

function packageMatlabCode(bOmitExternalLibraries)
    if(~exist('bOmitExternalLibraries'))
        bOmitExternalLibraries = true;
    end
    marshallingAreaBase = '/u/vvasuki/vishvas/work/software/';
    marshallingAreaMyCode = [marshallingAreaBase 'matlabPackage'];
    filePatternMyCode = 'vishvas/work/';
    filePatternLibraries = 'graft/matlabToolboxes';
    % archiveName = [marshallingArea '.zip'];
    
    filesString = path;
    files = StringUtilities.split(filesString, ':');

    function zipFiles(marshallingArea)
        [pathStr, dirName, ext] = fileparts(marshallingArea);
        cd(pathStr);
        system(['zip -r ' dirName ' ' dirName]);
    end
    IO.selectivelyGatherFiles(files, marshallingAreaMyCode, filePatternMyCode);
    zipFiles(marshallingAreaMyCode);

    if(~bOmitExternalLibraries)
        marshallingAreaLibraries = '/public/linux/graft/matlabToolboxes';
        zipFiles(marshallingAreaLibraries);
        system(['cp ' marshallingAreaLibraries '.zip ' marshallingAreaBase]);
    end
end


function testClass
    display 'Class definition is ok';
end

end
end