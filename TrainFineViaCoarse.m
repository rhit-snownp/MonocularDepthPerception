%Train the Fine Network on the Output of the coarse network


clc; 
close all; 
clear variables; 

%%
%load in the training datasets
[trainCombined, valCombined] = ReadDIODEforCombined("images\train\indoors\");

%%
%Load in the preexisting and pretrained networks
load('Fine Network 5.mat');
fineNet = net;
load('Coarse Network 1.mat');
coarseNet = net;
coarseNet.Layers(1:end-1);

%%
lgraph = layerGraph;
lgraph = addLayers(lgraph, freezeWeights(coarseNet.Layers(1:end-1)));
lgraph = addLayers(lgraph, fineNet.Layers(1:8));



%%
lgraph = connectLayers(lgraph, 'input', 'Fine 1');
lgraph = connectLayers(lgraph, 'reshape 1', 'Fine 2, Concat/in2');
analyzeNetwork(lgraph);



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
save("Combined Network 2",'net');


