clc; close all; clear variables; 
%%
load("combinedNet4.mat");
testCombined = ReadTestData("images\test");

%% Calculate threshold metric
out = exp(predict(net, testCombined));
target_depth_cells = readall(testCombined.UnderlyingDatastores{2});
test_data_length = length(target_depth_cells);
target_depths = reshape(cat(3,target_depth_cells{:}),[76 57 test_data_length]);
out = squeeze(out);
predictions = zeros(size(out));

for i=1:size(out,3)
    p95 = prctile(reshape(target_depths(:,:,i),1,[]),95);
    p5 = prctile(reshape(target_depths(:,:,i),1,[]),5);
    predictions(:,:,i) = rescale(out(:,:,i), p5, p95);
end


%% 
sigma_125 = calculate_threshold_metric(out, target_depths, 1.25)
sigma_125_2 = calculate_threshold_metric(out, target_depths, 1.25^2)
sigma_125_3 = calculate_threshold_metric(out, target_depths, 1.25^3)

%% Test individual images

images = testCombined.read();
inputImg = images{1};
groundTruth = images{2};

out = exp(predict(net, inputImg));
rescaledOutput = rescale(out, prctile(groundTruth(:),5),prctile(groundTruth(:),95));
sigma = calculate_threshold_metric(out, groundTruth, 1.25)
sigma2 = calculate_threshold_metric(rescaledOutput, groundTruth, 1.25)
subplot(2,2,4); imagesc(rescaledOutput);colorbar;axis equal;
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