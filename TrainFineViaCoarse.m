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
load('Transfer Learning Via Alexnet 2.mat');
coarseNet = net;

%%
lgraph = layerGraph;
lgraph = addLayers(lgraph, freezeWeights(coarseNet.Layers(1:end-1)));
lgraph = addLayers(lgraph, fineNet.Layers(1:8));



%%
% lgraph = connectLayers(lgraph, 'input', 'Fine 1');
% lgraph = connectLayers(lgraph, 'reshape 1', 'Fine 2, Concat/in2');


lgraph = connectLayers(lgraph, 'resize227', 'Fine 1');
lgraph = connectLayers(lgraph, 'resize-output-size', 'Fine 2, Concat/in2');
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
save("Combined Network 3 - Fine Training On Coarse Alexnet",'net');



%%
YPred = squeeze(predict(net,valCombined));


limit = 10;
for index = 1:limit

    inputImages = read(valCombined);
    figure;
    subplot(2,2,1);
    imshow(inputImages{1});
    title("Input Image");
    subplot(2,2,2);
    imagesc(inputImages{2});
    title("Depth Image");
    subplot(2,2,3);
    imagesc(YPred(:,:,index));
    title("Output Depth Map");
end


