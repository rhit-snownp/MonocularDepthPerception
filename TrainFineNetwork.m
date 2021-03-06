%Train the fine network


clc; 
close all; 
clear variables; 

%Read in the training data
[trainCombined, valCombined] = ReadDIODEforFineTraining("images\train\indoors\");


%%
%Orginal Pipeline
lgraph = pipeline1();

%Empty Layer Graph
myGraph = layerGraph;

%Grab off the input layer
inputLayer = lgraph.Layers(1);

%Generate the coarse layer input
SecondInputLayer = imageInputLayer([76 57 1],'Name','coarseImageInput'); 

%Relevant Layers from the fine network
fineLayers = lgraph.Layers(18:25);

%Connections
layers = addLayers(myGraph,fineLayers);
layers = addLayers(layers,inputLayer);
layers = addLayers(layers,SecondInputLayer);

%Connect the layers
layers = connectLayers(layers,'input','Fine 1');
layers = connectLayers(layers,'coarseImageInput','Fine 2, Concat/in2');




analyzeNetwork(layers);
%% Setup Training, and perform training
%     'ValidationData', valCombined, ...
%     'ValidationFrequency',3, ...
options = trainingOptions("sgdm", ...
    'MiniBatchSize',64, ...
    'MaxEpochs',20, ...
    'InitialLearnRate',1e-3, ...
    'Shuffle','every-epoch', ...
    'ValidationData', valCombined, ...
    'ValidationFrequency',25, ...
    'Verbose', false, ...
    'Plots','training-progress');

net = trainNetwork(trainCombined,layers,options);


%% Show pretty pictures off of the validation set
YPred = squeeze(predict(net,valCombined));

save('Fine Network 5','net');
limit = 10;
for index = 1:limit

    inputImages = read(valCombined);
    figure;
    subplot(2,2,1);
    imshow(inputImages{1});
    title("Input Image");
    subplot(2,2,2);
    imagesc(inputImages{2});
    title("Blurred Depth Image");
    subplot(2,2,3);
    imagesc(inputImages{3});
    title("Depth Image");
    subplot(2,2,4);
    imagesc(YPred(:,:,index));
    title("Upscaled Depth Image Image");
end


