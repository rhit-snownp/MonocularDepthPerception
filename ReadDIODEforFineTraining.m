function [trainCombined, valCombined] = ReadDIODEforFineTraining(relativePath)
    inputDataImages = imageDatastore(relativePath,"ReadFcn", @loadImage,"IncludeSubfolders",true);
    inputDataDepths = imageDatastore(relativePath, 'ReadFcn',@loadDIODEZDepthBlur,'FileExtensions','.npy',"IncludeSubfolders",true);
    outputDataDepths = imageDatastore(relativePath, 'ReadFcn',@loadDIODEZDepth,'FileExtensions','.npy',"IncludeSubfolders",true);


    n = length(outputDataDepths.Files);
    idx = randperm(n,round(n/5));

    valDataImages = subset(inputDataImages, idx);
    valDataDepths = subset(inputDataDepths, idx);
    valOutputDataDepths = subset(outputDataDepths, idx);
    idx = setdiff(1:n,idx); %invert the indicies

    trainDataImages = subset(inputDataImages, idx);
    trainDataDepths = subset(inputDataDepths, idx);
    trainOutputDataDepths = subset(outputDataDepths, idx);


    trainCombined = combine(trainDataImages, trainDataDepths,valOutputDataDepths);
    valCombined = combine(valDataImages, valDataDepths,trainOutputDataDepths);

    function data = loadDIODEZDepth(filename)
        addpath npy-matlab\
        data = readNPY(filename);
        data = imresize(data,[76,57]);
    end

    function data = loadImage(filename)
        im = imread(filename);
        data = imresize(im, [304 228]);
    end

function data = loadDIODEZDepthBlur(filename)
        addpath npy-matlab\
        data = readNPY(filename);
        data = imresize(data,[76,57]);
        %Blur the image output so we can train a result to upscale it
        data = imgaussfilt(data,2);
    end

end