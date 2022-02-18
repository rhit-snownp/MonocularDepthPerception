function [trainCombined, valCombined] = ReadDIODEToDatastore(relativePath)
    inputDataImages = imageDatastore(relativePath,"ReadFcn", @loadImage,"IncludeSubfolders",true);
    inputDataDepths = imageDatastore(relativePath, 'ReadFcn',@loadDIODEZDepth,'FileExtensions','.npy',"IncludeSubfolders",true);

    %split into training and validataion
    n = length(inputDataDepths.Files);
    idx = randperm(n,round(n/5));
    valDataImages = subset(inputDataImages, idx);
    valDataDepths = subset(inputDataDepths, idx);
    idx = setdiff(1:n,idx); %invert the indicies
    trainDataImages = subset(inputDataImages, idx);
    trainDataDepths = subset(inputDataDepths, idx);

    trainCombined = combine(trainDataImages, trainDataDepths);
    valCombined = combine(valDataImages, valDataDepths);

    %data augmentation
    trainCombined = transform(trainCombined, @imageRegressionAugmentationPipeline);
    valCombined = transform(valCombined, @imageRegressionAugmentationPipeline);

    function data = loadDIODEZDepth(filename)
        addpath npy-matlab\
        data = readNPY(filename);
        data = imresize(data,[76,57]);
    end

    function data = loadImage(filename)
        im = imread(filename);
        im = imgaussfilt(im,2);
        data = imresize(im, [304 228]);
    end

    function dataOut = imageRegressionAugmentationPipeline(dataIn)
    
    dataOut = cell([size(dataIn,1),2]);
    for i = 1:size(dataIn,1)
        
        inputImage = dataIn{i,1};
        targetImage = dataIn{i,2};
        
        % Add randomized rotation and scale and mirroring
        tform = randomAffine2d('Scale',[0.9,1.1],'Rotation',[-10 10],'XReflection',true);
        outputViewInput = affineOutputView(size(inputImage),tform);
        outputViewTarget = affineOutputView(size(targetImage),tform);

        % Use imwarp with the same tform and outputView to augment both images
        % the same way
        inputImageWarped = imwarp(inputImage,tform,'OutputView',outputViewInput);
        targetImageWarped = imwarp(targetImage,tform,'OutputView',outputViewTarget);
        
        dataOut(i,:) = {inputImageWarped,targetImageWarped};
    end

end

end