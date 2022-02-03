clc; 
close;
clear; 


load('TrainedNetwork1.mat')
[trainCombined, valCombined] = ReadDIODEToDatastore("images\train\indoors\");
layers = pipeline1();
% 
% 
% options = trainingOptions("sgdm", ...
%     'MiniBatchSize',32, ...
%     'MaxEpochs',4, ...
%     'InitialLearnRate',1e-4, ...
%     'Shuffle','every-epoch', ...
%     'ValidationData', valCombined, ...
%     'ValidationFrequency',10, ...
%     'Verbose', false, ...
%     'Plots','training-progress');
% 
% netTransfer = trainNetwork(trainCombined,layers,options);



%%

YPred = predict(netTransfer,(valCombined));
