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


%%
limit = 10;
for index = 1:limit

    inputImages = read(valCombined);
    figure;
    subplot(1,2,1);
    imshow(inputImages{1});
    title("Input Image");
    subplot(1,2,2);

    imagesc(inputImages{2});
    title("Depth Image");
    
end