

function [trainCombined, valCombined] = ReadDIODEforFineTraining(relativePath)
%% Function for reading in DIODE images and depths, as well as a psuedo course input to the system

    %Read each into the datastore with a custom function for each
    inputDataImages = imageDatastore(relativePath,"ReadFcn", @loadImage,"IncludeSubfolders",true);
    inputDataDepths = imageDatastore(relativePath, 'ReadFcn',@loadDIODEZDepthBlur,'FileExtensions','.npy',"IncludeSubfolders",true);
    outputDataDepths = imageDatastore(relativePath, 'ReadFcn',@loadDIODEZDepth,'FileExtensions','.npy',"IncludeSubfolders",true);
    

    %Split up the  data into train and validation datasets
    n = length(outputDataDepths.Files);
    idx = randperm(n,round(n/5));

    valDataImages = subset(inputDataImages, idx);
    valDataDepths = subset(inputDataDepths, idx);
    valOutputDataDepths = subset(outputDataDepths, idx);
    idx = setdiff(1:n,idx); %invert the indicies

    trainDataImages = subset(inputDataImages, idx);
    trainDataDepths = subset(inputDataDepths, idx);
    trainOutputDataDepths = subset(outputDataDepths, idx);

    %Make them into combined datastores
    trainCombined = combine(trainDataImages, trainDataDepths,trainOutputDataDepths);
    valCombined = combine(valDataImages, valDataDepths,valOutputDataDepths);


    %% Custom reading in functions for depth and images
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
        readdata = readNPY(filename);
        
       %Use bilinear upscaleing 8x to make a much more fuzzy input image of
       %the depth
        scale = 8;
        downscaledData = imresize(readdata, 1/scale);
        poorlyUpscaledDate = imresize(downscaledData,scale);
        data = imresize(poorlyUpscaledDate,[76,57]);
        
        %Gaussian blur for good measure
        data = imgaussfilt(data);
    end

end