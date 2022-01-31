clc
clear all
close all

%Test Training of Pipeline 1 on NYU Depth Dataset


%Load in the NYU Depth Dataset
load('NYU Dataset\nyu_depth_v2_labeled.mat')

%%
% trainX = images(:,:,:,1:round(1449/2));
trainY = depths(:,:,1:round(1449/2));
% testX = images(:,:,:,round(1449/2)+1:end);
testY = depths(:,:,round(1449/2)+1:end);
newTrainY = repmat(reshape(trainY, size(trainY,1), size(trainY,2), 1, size(trainY,3)), 1, 1, 3, 1);
newTestY = repmat(reshape(testY, size(testY,1), size(testY,2), 1, size(testY,3)), 1, 1, 3, 1);

trainX = imageDatastore("NYU Dataset/Training Data/Input/",'LabelSource','foldernames');
% trainY = imageDatastore("NYU Dataset/Training Data/Output/",'LabelSource','foldernames');
testX = imageDatastore("NYU Dataset/Testing Data/Input/",'LabelSource','foldernames');
% testY = imageDatastore("NYU Dataset/Testing Data/Output/",'LabelSource','foldernames');




%Rescale input data to 304×228x1
imageSize = uint8([304 228 1]);
augmentedTrainImages = augmentedImageDatastore(imageSize,trainX,trainY,'ColorPreprocessing','rgb2gray');
augmentedTestImages = augmentedImageDatastore(imageSize,testX,testY,'ColorPreprocessing','rgb2gray');
%Load in the defined Pipeline 1 Network Architecture

[layers] = pipeline1();


options = trainingOptions('sgdm', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-3, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false);
%%
net = trainNetwork(trainX,newTrainY,layers,options);