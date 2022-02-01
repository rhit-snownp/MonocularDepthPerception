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

    % fixes a "known" issue with combining datastores acording to matlab...
%     valDataImages = transform(valDataImages, @(x){x});
%     valDataDepths = transform(valDataDepths, @(x){x});
%     trainDataImages = transform(trainDataImages, @(x){x});
%     trainDataDepths = transform(trainDataDepths, @(x){x});

    trainCombined = combine(trainDataImages, trainDataDepths);
    valCombined = combine(valDataImages, valDataDepths);

    function data = loadDIODEZDepth(filename)
        addpath npy-matlab\
        data = readNPY(filename);
%         depthMat(:,:,i) = data;
%         data = repmat(data, [1,1,3]);
        data = imresize(data,[76,57]);
    end

    function data = loadImage(filename)
        im = imread(filename);
        data = imresize(im, [304, 228]);
    end

end