
function [lgraph] = pipeline1()

%read in the data 
lgraph = layerGraph;

coarseNetwork = [
    %input 
    imageInputLayer([304 228 3],'Name','input')  
    %coarse 1 
    convolution2dLayer([11 11],96,'Stride',[4 4],'Name','coarse 1')
    reluLayer("Name",'relu 1')
    maxPooling2dLayer([2 2],'Name','Pool 1','Stride',[2,2])
    %coarse 2 
    convolution2dLayer([5 5],256,'Name','coarse 2')
    reluLayer("Name",'relu 2')
    maxPooling2dLayer([2 2],'Name','Pool 2','Stride',[2,2])
    %coarse 3
    convolution2dLayer([3 3],384,'Name','coarse 3')
    reluLayer("Name",'relu 3')
%     maxPooling2dLayer([2 2],'Name','Pool 3','Stride',[2,2])
    %coarse 4 
    convolution2dLayer([3 3],384,'Name','coarse 4')
    reluLayer("Name",'relu 4')
%     maxPooling2dLayer([2 2],'Name','Pool 4','Stride',[2,2])
    %coarse 5 
    convolution2dLayer([3 3], 256,'Name','coarse 5')
    reluLayer("Name",'relu 5')
    %FC 1 
    fullyConnectedLayer(4096,'Name','FC 1')
    reluLayer("Name", 'relu 6')
    %FC 2 
    %not positive on the output size of this one. maybe jsut the dimensions
    %of the actual input image trimmed down, since it looks like that in
    %the paper
    fullyConnectedLayer(4332,'Name','FC 2')
    reluLayer("Name", 'relu 7')
    reshapeLayer('reshape 1')
    ];
    
lgraph = addLayers(lgraph,coarseNetwork);

fineNetworkPart1 = [
    %input 2
    imageInputLayer([304 228],'Name','input 2') 
    %Fine 1
    convolution2dLayer([9 9],63,'Stride',[4 4],'Name','Fine 1','Padding',3)
    reluLayer("Name",'relu fine 1')
    maxPooling2dLayer([2 2],'Name','Pool Fine', 'Padding','same')
    ];

lgraph = addLayers(lgraph,fineNetworkPart1);

concat = depthConcatenationLayer(2,'Name','Fine 2, Concat');
lgraph = addLayers(lgraph,concat);

lgraph = connectLayers(lgraph,'reshape 1','Fine 2, Concat/in1');
lgraph = connectLayers(lgraph,'Pool Fine','Fine 2, Concat/in2');

fineNetworkPart2 = [
    %Fine 3
    convolution2dLayer([5 5],64,'Name','Fine 3','Padding',2)
    reluLayer("Name",'relu fine 3')
    %Fine 4
    convolution2dLayer([5 5],1,'Name','Fine 4','Padding',2)
    %If we said that the depth/how many filters you have is the "64"/ the
    %arrow number, then what is it for the concatenation layer?
    SIERegressionLayer("Scale-Invarient Error")
];
lgraph = addLayers(lgraph,fineNetworkPart2);

lgraph = connectLayers(lgraph,'Fine 2, Concat','Fine 3');
% plot(lgraph)


% figure
% plot(lgraph)
