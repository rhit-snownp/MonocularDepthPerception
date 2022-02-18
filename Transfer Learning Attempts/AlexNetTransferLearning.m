%ALEXNET Transfer Learning Experiment

clc; 
close all; 
clear variables; 

%load in the training datasets
[trainCombined, valCombined] = ReadDIODEforCombined("..\images\train\indoors\");

net = alexnet;
analyzeNetwork(net);

%%  Define the Modified Network Architecture

%input layer 
top1 = imageInputLayer([304 228 3],'Name','input'); 

%Reshape layer to get to 227 x 227 x 3
top2 = resize3dLayer('OutputSize',[227 227 3],'Name','resize227');

%Grab most of the layers of alexnet we care about
middle1 = net.Layers(2:end-5);

%Generate a new tail to be a regression layer 
end1 = [
depthToSpace2dLayer([64 64],"Name","depthToSpace","Mode","crd")
resize2dLayer("Name","resize-output-size","GeometricTransformMode","half-pixel","Method","nearest","NearestRoundingMode","round","OutputSize",[76 57])
regressionLayer('Name','FinalRegressionLayer')];

%Assemble the new network
lgraph = layerGraph;
lgraph = addLayers(lgraph, top1);
lgraph = addLayers(lgraph, top2);
lgraph = addLayers(lgraph, middle1);
lgraph = addLayers(lgraph, end1);

lgraph = connectLayers(lgraph, 'input', 'resize227');
lgraph = connectLayers(lgraph, 'resize227', 'conv1');
lgraph = connectLayers(lgraph, 'fc7', 'depthToSpace');

analyzeNetwork(lgraph);


%%
options = trainingOptions("adam", ...
    'MiniBatchSize',32, ...
    'MaxEpochs',15, ...
    'InitialLearnRate',1e-5, ...
    'Shuffle','every-epoch', ...
    'ValidationData', valCombined, ...
    'ValidationFrequency',50, ...
    'Verbose', false, ...
    'Plots','training-progress');

net = trainNetwork(trainCombined,lgraph,options);
save("Transfer Learning Via Alexnet 2",'net');



%% Testing if network already exists
load("..\Trained Networks\Transfer Learning Via Alexnet 2.mat")
[trainCombined, valCombined] = ReadDIODEforCombined("..\images\train\indoors\");
[testCombined] = ReadTestData("..\images\test\indoors\");

%% Test on Cleaned Testing Set
YPred = squeeze(predict(net,testCombined));
reset(testCombined)
limit = 100;
for index = 1:limit

    inputImages = read(testCombined);
    figure;
    subplot(2,2,1);
    imshow(inputImages{1});
    title("Input Image");
    subplot(2,2,2);
    imagesc(inputImages{2});
    title("Depth Image");

    %Calculate the statistics on how good or bad the output image is:
    exponentialResult = YPred(:,:,index);
    groundTruth = inputImages{2};

    sigma_125 = calculate_threshold_metric(exponentialResult, groundTruth, 1.25);
    sigma_125_2 = calculate_threshold_metric(exponentialResult, groundTruth, 1.25^2);
    sigma_125_3 = calculate_threshold_metric(exponentialResult, groundTruth, 1.25^3);

    subplot(2,2,3);
    imagesc(YPred(:,:,index));
    title({"Output Depth Map:", sprintf("\\sigma<1.25=%g3%%,  \\sigma<1.25^2=%g3%%, \\sigma<1.25^3=%g3%%",sigma_125*100,sigma_125_2*100,sigma_125_3*100)});
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