% reads data for training the combined network
function [trainCombined, valCombined] = ReadDIODEforCombined(relativePath)
    inputDataImages = imageDatastore(relativePath,"ReadFcn", @loadImage,"IncludeSubfolders",true);
    outputDataDepths = imageDatastore(relativePath, 'ReadFcn',@loadDIODEZDepth,'FileExtensions','.npy',"IncludeSubfolders",true);
    

    %Split up the images into train and validation sets (80/20 split)
    n = length(outputDataDepths.Files);
    idx = randperm(n,round(n/5));

    valDataImages = subset(inputDataImages, idx);
    valOutputDataDepths = subset(outputDataDepths, idx);
    idx = setdiff(1:n,idx); %invert the indicies

    trainDataImages = subset(inputDataImages, idx);
    trainOutputDataDepths = subset(outputDataDepths, idx);


    trainCombined = combine(trainDataImages, trainOutputDataDepths);
    valCombined = combine(valDataImages, valOutputDataDepths);

    function data = loadDIODEZDepth(filename)
        addpath npy-matlab\
        data = readNPY(filename);
        data = imresize(data,[76,57]);
    end

    function data = loadImage(filename)
        im = imread(filename);
        data = imresize(im, [304 228]);
    end

end