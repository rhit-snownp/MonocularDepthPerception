clc; close all; clear variables; 

[inputDataDepths,inputDataImages] = ReadDIODEToDatastore("images\train\indoors\");

lgraph = pipeline1();
inputSize = lgraph.Layers(1).InputSize;
coarseGraph = lgraph.Layers(1:16);

patchds = randomPatchExtractionDatastore(inputDataImages,inputDataDepths,inputSize,'PatchesPerImage',5);
idx = randperm(patchds.NumObservations,round(patchds.NumObservations/3));
valData = partitionByIndex(patchds, idx);
trainData = partitionByIndex(patchds, setdiff(1:patchds.NumObservations,idx));

layers = [coarseGraph
          resizeLayer('resize')
          SIERegressionLayer("SIE")];
%%
options = trainingOptions("sgdm", ...
    'MiniBatchSize',128, ...
    'MaxEpochs',4, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData', valData, ...
    'ValidationFrequency',3, ...
    'Verbose', false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(trainData,layers,options);
