clc
clear
close all

%Test Training of Pipeline 1 on NYU Depth Dataset


%Load in the NYU Depth Dataset
%load('NYU Dataset\nyu_depth_v2_labeled.mat')

%%
% % trainX = images(:,:,:,1:round(1449/2));
% trainY = depths(:,:,1:round(1449/2));
% % testX = images(:,:,:,round(1449/2)+1:end);
% testY = depths(:,:,round(1449/2)+1:end);
% newTrainY = repmat(reshape(trainY, size(trainY,1), size(trainY,2), 1, size(trainY,3)), 1, 1, 3, 1);
% newTestY = repmat(reshape(testY, size(testY,1), size(testY,2), 1, size(testY,3)), 1, 1, 3, 1);

trainX = imageDatastore("NYU Dataset/Training Data/Input/",'LabelSource','foldernames',"ReadFcn", @loadImage);
trainY = imageDatastore("NYU Dataset/Training Data/Output/",'LabelSource','foldernames',"ReadFcn", @loadImage);
testX = imageDatastore("NYU Dataset/Testing Data/Input/",'LabelSource','foldernames',"ReadFcn", @loadImage);
testY = imageDatastore("NYU Dataset/Testing Data/Output/",'LabelSource','foldernames',"ReadFcn", @loadImage);






%Rescale input data to 304x228x1
% imageSize = uint8([304 228 1]);
% augmentedTrainImages = augmentedImageDatastore(imageSize,trainX,trainY,'ColorPreprocessing','rgb2gray');
% augmentedTestImages = augmentedImageDatastore(imageSize,testX,testY,'ColorPreprocessing','rgb2gray');
%Load in the defined Pipeline 1 Network Architecture


imageSize = uint8([304 228 1]);
augmentedTrainX = augmentedImageDatastore(imageSize,trainX,'ColorPreprocessing','rgb2gray');
augmentedTrainY = augmentedImageDatastore(imageSize,trainY);

%dsTrain = combine(augmentedTrainX,augmentedTrainX,augmentedTrainY);
dsTrain = combine(trainX,trainY,trainY);



[layers] = pipeline1();
analyzeNetwork(layers)

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

net = trainNetwork(dsTrain,layers,options);



function data = loadImage(filename)
        im = imread(filename);
        data = imresize(im, [304, 228]);
end

