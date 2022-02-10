clc; close all; clear variables; 
%%
% load("combinedNet.mat");
testCombined = ReadTestData("images\test");

%% Calculate threshold metric
out = exp(predict(combinedNet, trainCombined));
target_depth_cells = readall(trainCombined.UnderlyingDatastores{2});
test_data_length = length(target_depth_cells);
target_depths = reshape(cat(3,target_depth_cells{:}),[76 57 1 test_data_length]);
%% 
sigma_125 = calculate_threshold_metric(out, target_depths, 1.25)
sigma_125_2 = calculate_threshold_metric(out, target_depths, 1.25^2)
sigma_125_3 = calculate_threshold_metric(out, target_depths, 1.25^3)

%% Test individual images

images = testCombined.read();
inputImg = images{1};
% inputImg = ones([227 227 3]);
out = exp(predict(combinedNet, inputImg));
subplot(2,2,3); imshow(inputImg);
subplot(2,2,2); imagesc(out); title("output depthmap");colorbar;axis equal;
subplot(2,2,1); imagesc(images{2}); title("ground truth");colorbar; axis equal;


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