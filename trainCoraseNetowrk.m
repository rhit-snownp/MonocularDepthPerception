clc; close all; clear variables; 

[trainCombined, valCombined] = ReadDIODEToDatastore("images\train\indoors\");
%%
lgraph = pipeline1();
inputSize = lgraph.Layers(1).InputSize;
coarseGraph = lgraph.Layers(1:17);
% a = alexnet;
% alexnetTransfer = a.Layers(1:end-3);
% layers = [ alexnetTransfer
%             fullyConnectedLayer(4332,'Name','FC 2')
%             reluLayer("Name", 'relu 7')
%             reshapeLayer('reshape 1')
%           SIERegressionLayer("SIE")];

layers = [ coarseGraph
          SIERegressionLayer("SIE")];
%%
%     'ValidationData', valCombined, ...
%     'ValidationFrequency',3, ...
options = trainingOptions("sgdm", ...
    'MiniBatchSize',32, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', valCombined, ...
    'ValidationFrequency',50, ...
    'Verbose', false, ...
    'Plots','training-progress');

net = trainNetwork(trainCombined,layers,options);
