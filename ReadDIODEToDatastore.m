function [trainCombined, valCombined] = ReadDIODEToDatastore(relativePath)
    inputDataImages = imageDatastore(relativePath,"ReadFcn", @loadImage,"IncludeSubfolders",true);
%     augDataImages = augmentedImageDatastore([304, 228], inputDataImages);
    inputDataDepths = imageDatastore(relativePath, 'ReadFcn',@loadDIODEZDepth,'FileExtensions','.npy',"IncludeSubfolders",true);
%     augDataDepths = augmentedImageDatastore([76,57], inputDataDepths);

    n = length(inputDataDepths.Files);
    idx = randperm(n,round(n/5));
%     valDataImages = partitionByIndex(augDataImages, idx);
    valDataImages = subset(inputDataImages, idx);
    valDataDepths = subset(inputDataDepths, idx);
    idx = setdiff(1:n,idx); %invert the indicies
%     trainDataImages = partitionByIndex(augDataImages, idx);
    trainDataImages = subset(inputDataImages, idx);
    trainDataDepths = subset(inputDataDepths, idx);

    trainCombined = combine(trainDataImages, trainDataDepths);
    valCombined = combine(valDataImages, valDataDepths);

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
        
        % Resize images to 32-by-32 pixels and convert to data type single
        inputImage = im2single(dataIn{i,1});
        targetImage = dataIn{i,2};
        
        % Add salt and pepper noise
%         inputImage = imnoise(inputImage,'salt & pepper');
        
        % Add randomized rotation and scale
        tform = randomAffine2d('Scale',[0.9,1.1],'Rotation',[-10 10],'XReflection',true);
        outputViewInput = affineOutputView(size(inputImage),tform);
        outputViewTarget = affineOutputView(size(targetImage),tform);

        % Use imwarp with the same tform and outputView to augment both images
        % the same way
        inputImageWarped = imwarp(inputImage,tform,'OutputView',outputViewInput);
        targetImageWarped = imwarp(targetImage,tform,'OutputView',outputViewTarget);
        
%         dataOut(2*i-1,:) = {inputImageWarped,targetImageWarped};
%         dataOut(2*i,:) = {inputImage,targetImage};
        dataOut(i,:) = {inputImageWarped,targetImageWarped};
    end

end

end