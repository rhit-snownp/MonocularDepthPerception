%This file runs the network on the test set and calculates thresholds
clc; close all; clear variables; 
%% Load network and test data
load("combinedNet4.mat");
testCombined = ReadTestData("images\test");

%% Pass test set through network
out = exp(predict(net, testCombined));
target_depth_cells = readall(testCombined.UnderlyingDatastores{2}); %get the ground truth data
test_data_length = length(target_depth_cells);
target_depths = reshape(cat(3,target_depth_cells{:}),[76 57 test_data_length]);
out = squeeze(out);
predictions = zeros(size(out));


%%
u = 95; %upper precentile for scaling
l = 100-u; %lower

%Rescale the network output so that 5th and 95th percentile match the
%ground truth. Save that data in predictions
for i=1:size(out,3)
    outDepth = out(:,:,i);
    targetmax = prctile(reshape(target_depths(:,:,i),1,[]),u);
    targetmin = prctile(reshape(target_depths(:,:,i),1,[]),l);
    outputmax = prctile(reshape(outDepth,1,[]),u);
    outputmin = prctile(reshape(outDepth,1,[]),l);
    predictions(:,:,i) = targetmin + [(outDepth-outputmin)./(outputmax-outputmin)].*(targetmax-targetmin);
end


%% 
% delta values before scaling
disp("before scaling")
delta_125 = calculate_threshold_metric(out, target_depths, 1.25)
delta_125_2 = calculate_threshold_metric(out, target_depths, 1.25^2)
delta_125_3 = calculate_threshold_metric(out, target_depths, 1.25^3)

%delta values after scaling
disp("after scaling")
delta_125 = calculate_threshold_metric(predictions, target_depths, 1.25)
delta_125_2 = calculate_threshold_metric(predictions, target_depths, 1.25^2)
delta_125_3 = calculate_threshold_metric(predictions, target_depths, 1.25^3)
%% Test individual images

images = testCombined.read();
inputImg = images{1};
groundTruth = images{2};

out = exp(predict(net, inputImg));
rescaledOutput = rescale(out, prctile(groundTruth(:),l),prctile(groundTruth(:),u));
delta = calculate_threshold_metric(out, groundTruth, 1.25)
delta2 = calculate_threshold_metric(rescaledOutput, groundTruth, 1.25)
subplot(2,2,4); imagesc(rescaledOutput);title("Scaled Output Depth Map \delta < 1.25:",sprintf(" %1.3f",delta2));colorbar;axis equal;
subplot(2,2,1); imshow(inputImg); title("Input Image");
subplot(2,2,3); imagesc(out); title("Output Depth Map \delta < 1.25:",sprintf(" %1.3f",delta));colorbar;axis equal;
subplot(2,2,2); imagesc(groundTruth); title("Depth Image");colorbar; axis equal;


function [testCombined] = ReadTestData(relativePath)
    inputDataImages = imageDatastore(relativePath,"ReadFcn", @loadImage,"IncludeSubfolders",true);
    inputDataDepths = imageDatastore(relativePath, 'ReadFcn',@loadDIODEZDepth,'FileExtensions','.npy',"IncludeSubfolders",true);

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