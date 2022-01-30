%read in the data 
lgraph = layerGraph;

coarseNetwork = [
    %input 
    imageInputLayer([304 228],'Name','input')  
    %coarse 1 
    convolution2dLayer([11 11],96,'Stride',[4 4],'Name','coarse 1')
    maxPooling2dLayer([2 2],'Name','Pool 1')
    %coarse 2 
    convolution2dLayer([5 5],256,'Name','coarse 2')
    maxPooling2dLayer([2 2],'Name','Pool 2')
    %coarse 3
    convolution2dLayer([3 3],384,'Name','coarse 3')
    %coarse 4 
    convolution2dLayer([3 3],384,'Name','coarse 4')
    %coarse 5 
    convolution2dLayer([3 3],256,'Name','coarse 5')
    %FC 1 
    fullyConnectedLayer(4096,'Name','FC 1')
    %FC 2 
    %not positive on the output size of this one. maybe jsut the dimensions
    %of the actual input image trimmed down, since it looks like that in
    %the paper
    fullyConnectedLayer(4096,'Name','FC 2')
    ]
    
lgraph = addLayers(lgraph,coarseNetwork);

fineNetworkPart1 = [
    %Fine 1
    convolution2dLayer([9 9],63,'Stride',[4 4],'Name','Fine 1')
    maxPooling2dLayer([2 2],'Name','Pool Fine')

    ]

lgraph = addLayers(lgraph,fineNetworkPart1);

concat = depthConcatenationLayer(2,'Name','Fine 2, Concat')
lgraph = addLayers(lgraph,concat);

lgraph = connectLayers(lgraph,'FC 2','Fine 2, Concat/in1');
lgraph = connectLayers(lgraph,'Pool Fine','Fine 2, Concat/in2');

fineNetworkPart2 = [
    %Fine 3
    convolution2dLayer([5 5],64,'Name','Fine 3')
    %Fine 4
    convolution2dLayer([5 5],64,'Name','Fine 4')
    %If we said that the depth/how many filters you have is the "64"/ the
    %arrow number, then what is it for the concatenation layer? 
]
lgraph = addLayers(lgraph,fineNetworkPart2);

lgraph = connectLayers(lgraph,'Fine 2, Concat','Fine 3');
plot(lgraph)


figure
plot(lgraph)