clc; clear variables; close all;

[trainCombined, valCombined] = ReadDIODEforCombined("images\train\indoors\");

load('Fine Network 5.mat');
fineNet = net;
load('Coarse Network 1.mat');
coarseNet = net;
lgraph = layerGraph;

lgraph = addLayers(lgraph, coarseNet.Layers(1:end-1));
lgraph = addLayers(lgraph, fineNet.Layers(1:8));
%%
lgraph = connectLayers(lgraph, 'input', 'Fine 1');
lgraph = connectLayers(lgraph, 'reshape 1', 'Fine 2, Concat/in2');
%%
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

%%
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




%% Calculate threshold metrics
load("Combined Network 1")
testCombined = ReadTestData("images\val");
out = exp(predict(net, testCombined));
target_depth_cells = readall(testCombined.UnderlyingDatastores{2});
test_data_length = length(target_depth_cells);
target_depths = reshape(cat(3,target_depth_cells{:}),[76 57 1 test_data_length]);

%%
sigma_125 = calculate_threshold_metric(out, target_depths, 1.25)
sigma_125_2 = calculate_threshold_metric(out, target_depths, 1.25^2)
sigma_125_3 = calculate_threshold_metric(out, target_depths, 1.25^3)


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



   