clc
clear
close all

%Test Training of Pipeline 1 on NYU Depth Dataset


%Load in the NYU Depth Dataset
load('NYU Dataset\nyu_depth_v2_labeled.mat')

%%
% % trainX = images(:,:,:,1:round(1449/2));
% trainY = depths(:,:,1:round(1449/2));
% % testX = images(:,:,:,round(1449/2)+1:end);
% testY = depths(:,:,round(1449/2)+1:end);
% newTrainY = repmat(reshape(trainY, size(trainY,1), size(trainY,2), 1, size(trainY,3)), 1, 1, 3, 1);
% newTestY = repmat(reshape(testY, size(testY,1), size(testY,2), 1, size(testY,3)), 1, 1, 3, 1);

%Read in the input data images
trainX = imageDatastore("NYU Dataset/Training Data/Input/",'LabelSource','foldernames',"ReadFcn", @loadImage);
trainY = imageDatastore("NYU Dataset/Training Data/Output/",'LabelSource','foldernames',"ReadFcn", @loadDepthImage);
testX = imageDatastore("NYU Dataset/Testing Data/Input/",'LabelSource','foldernames',"ReadFcn", @loadImage);
testY = imageDatastore("NYU Dataset/Testing Data/Output/",'LabelSource','foldernames',"ReadFcn", @loadDepthImage);



%Combine the data into a combined datastore
dsTrain = combine(trainX,trainY);
dsTest = combine(testX,testY);

%Load the pipeline
[layers] = pipeline1();
%analyzeNetwork(layers)


%train the network
options = trainingOptions('sgdm', ...
    'MiniBatchSize',128, ...
    'MaxEpochs',30, ...
    'InitialLearnRate',1e-5, ...
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

function data = loadDepthImage(filename)
        im = imread(filename);
        data = imresize(im, [76, 57]);
end

