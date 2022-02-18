%% Script to combine a pre trained coarse and a pre trained fine network, and then train the 
clc; clear variables; close all;

[trainCombined, valCombined] = ReadDIODEforCombined("images\train\indoors\");

%% Load existing neural nets, and adjust the layers
load('Trained Networks\Fine Network 5.mat');
fineNet = net;
load('Trained Networks\coarseNet2.mat');
coarseNet = net;
lgraph = layerGraph;

lgraph = addLayers(lgraph, freezeWeights(coarseNet.Layers(1:end-1)));
lgraph = addLayers(lgraph, fineNet.Layers(1:8));

lgraph = connectLayers(lgraph, 'input', 'Fine 1');
lgraph = connectLayers(lgraph, 'reshape 1', 'Fine 2, Concat/in2');

%% Setup the Training Options
options = trainingOptions("adam", ...
    'MiniBatchSize',32, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', valCombined, ...
    'ValidationFrequency',50, ...
    'Verbose', false, ...
    'Plots','training-progress');

net = trainNetwork(trainCombined,lgraph,options);
save("Combined Network 1",'net');
YPred = squeeze(predict(net,valCombined));


%% Print some plots from the pre loaded validation set
limit = 10;

for index = 1:limit
    inputImages = read(valCombined);
    figure;
    subplot(2,2,1);
    imshow(inputImages{1});
    title("Input Image");
    subplot(2,2,3);
    imagesc(inputImages{2});
    title("Depth Image");
    subplot(2,2,4);
    imagesc(YPred(:,:,index));
    title("Output");
end




%% Function to Read Test Set
function [testCombined] = ReadTestData(relativePath)
    inputDataImages = imageDatastore(relativePath,"ReadFcn", @loadImage,"IncludeSubfolders",true);
%     augDataImages = augmentedImageDatastore([304, 228], inputDataImages);
    inputDataDepths = imageDatastore(relativePath, 'ReadFcn',@loadDIODEZDepth,'FileExtensions','.npy',"IncludeSubfolders",true);
%     augDataDepths = augmentedImageDatastore([76,57], inputDataDepths);

    testCombined = combine(inputDataImages, inputDataDepths);

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
end



   