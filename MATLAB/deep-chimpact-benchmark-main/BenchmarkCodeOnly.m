%% Deep Chimpact: Depth Estimation for Wildlife Conservation – MATLAB Benchmark Code

%% Load labels and metadata
labels = readtable('train_labels.csv');
head(labels)
metadata = readtable('train_metadata.csv');
head(metadata)

%% Access & Process Video Files
tempimds = fullfile(tempdir,"imds.mat");
if exist(tempimds,'file')
    load(tempimds,'imds')
else
    
    imds = imageDatastore('s3://drivendata-competition-depth-estimation-public/train_videos_downsampled/',...
                          'ReadFcn',@(filename)readVideo(filename,metadata,'TrainingData'),...
                          'FileExtensions',{'.mp4','.avi'});
    
    save(tempimds,"imds");
end
files = imds.Files;

rng(0); %Seed the random number generator for repeatability 
idx = randperm(numel(files));
imds = imds.subset(idx(1:100)); %Random Subset for testing

imds.readall("UseParallel", true);

%% Designing the Neural Network

%% Approach 1: Using Deep Network Designer
deepNetworkDesigner

%% Approach 2: Create network programmatically
lgraph_1 = layerGraph(resnet18);

inputLayers = [imageInputLayer([480 640],'Name','imageinput'),...
               resize3dLayer('OutputSize',[480 640 3],'Name','resize3D-output-size')]
lgraph_1 = replaceLayer(lgraph_1,'data',inputLayers);

lgraph_1 = removeLayers(lgraph_1,{'fc1000','prob','ClassificationLayer_predictions'});

outputLayers = [fullyConnectedLayer(1,'Name','fc'),regressionLayer('Name','regressionoutput')] 

lgraph_1 = addLayers(lgraph_1,outputLayers);
lgraph_1 = connectLayers(lgraph_1,'pool5','fc');

%% Analyze network
analyzeNetwork(lgraph_1)

%% Prepare Training Data
imageDS = imageDatastore('TrainingData');
labelDS = generateLabelDS(imageDS,labels);
fullDataset = combine(imageDS, labelDS);

n_images = numel(imageDS.Files);
n_training = round(0.8*n_images);
idx = randperm(n_images);

trainingIdx = idx(1:n_training);
validationIdx = idx(n_training+1:end);

trainingDS = subset(fullDataset,trainingIdx);
validationDS = subset(fullDataset,validationIdx);

%% Specify Training Options
miniBatchSize = 8;
validationFrequency = floor(n_training/miniBatchSize);

options = trainingOptions('sgdm', ...
    'MiniBatchSize',miniBatchSize, ...
    'MaxEpochs',50, ...
    'InitialLearnRate',0.01, ...
    'Shuffle','every-epoch', ...
    'ValidationData',validationDS, ...
    'ValidationFrequency',validationFrequency, ...
    'Plots','training-progress', ...
    'Verbose',true, ...
    'ExecutionEnvironment',"auto");

%% Train the Network
[net,info] = trainNetwork(trainingDS,lgraph_1,options);
save('trainedNetwork.mat',net)

%% Prepare Test Data
test_metadata = readtable('test_metadata.csv');

imds = imageDatastore('s3://drivendata-competition-depth-estimation-public/test_videos_downsampled/',...
                      'ReadFcn',@(filename)readVideo(filename,test_metadata,'TestData'),...
                      'FileExtensions',{'.mp4','.avi'});

imds.readall("UseParallel",true);

%% Depth Estimation Using Test Data

% Initialise output table
results = table('Size',[height(test_metadata) 3],'VariableTypes',{'string','string','single'},'VariableNames',{'video_id','time','distance'});

for i = 1:height(test_metadata)
    id = test_metadata.video_id{i};
    t = test_metadata.time(i);
    filename = [id(1:4) num2str(t) '.png']; %Find corresponding image name
    file = fullfile('TestData',filename);
    if isfile(file) %Frames without bounding boxes will not have input images
        I = imread(file);
        prediction = predict(net,I,'ExecutionEnvironment','auto');
    else
        prediction = 0;
    end    
    results.video_id(i) = id;
    results.time(i) = t;
    results.distance(i) = prediction;
end
head(results)

%% Save Submission to File
writetable(results,'Submission.csv');
%% Helper Functions

%% Video pre-processing - Optical Flow
function output = readVideo(filename, metadata, folder)
    %Load video
    vr = VideoReader(filename);
    H = vr.Height;
    W = vr.Width;
    
    [~, name] = fileparts(filename);
    idx = contains(metadata.video_id,name);
    
    videoMetadata = rmmissing(metadata(idx,:)); %Ignore frames without bounding box
    
    n_Frames = height(videoMetadata);
    
    %Preallocate the output image array
    output = zeros(480,640,n_Frames);
    
    for i = 1:n_Frames
        opticFlow = opticalFlowLK('NoiseThreshold',0.009); %Define optical flow
        t = videoMetadata.time(i); %Extract timestamp
        try
            if t == 0 %If first frame compare with second
                f1 = vr.read(1);
                f2 = vr.read(2);
            else %Otherwise take current frame (t+1) and previous (t)
                f1 = vr.read(t);
                f2 = vr.read(t+1);
            end
        catch
            continue %Ignore videos where timings don't match with frames
        end
        %Convert to grayscale
        f1Gray = im2gray(f1);
        f2Gray = im2gray(f2);
        
        %Calculate optical flow
        estimateFlow(opticFlow,f1Gray);
        flow = estimateFlow(opticFlow,f2Gray);
        
        %Extract corners of bounding box
        x1 = videoMetadata.x1(i);
        x2 = videoMetadata.x2(i);
        y1 = videoMetadata.y1(i);
        y2 = videoMetadata.y2(i);
        
        %Apply mask for bounding box
        mask = poly2mask([x1 x2 x2 x1]*W,[y1 y1 y2 y2]*H,H,W);
        maskedFlow = bsxfun(@times, flow.Magnitude, cast(mask, 'like', flow.Magnitude));

        maskedFlow = imresize(maskedFlow,'OutputSize',[480 640]);
        
        file = fullfile(folder, [name num2str(t) '.png']); %Generate file name
        
        %Save image to file
        if isfolder(folder)
            imwrite(maskedFlow,file)
        else
            mkdir(folder)
            imwrite(maskedFlow,file)
        end
        output(:,:,i) = maskedFlow;
    end
end

%% Create arrayDatastore for responses
function labelDS = generateLabelDS(imds,labels)
    files = imds.Files;
    n_files = length(files);
    dataLabels = zeros(n_files,1);
    for i = 1:n_files
        [~,id] = fileparts(files{i});
        video = id(1:4);
        time = str2double(id(5:end));
        idx = (contains(labels.video_id,video)) & (labels.time == time);
        dataLabels(i) = labels.distance(idx);
    end
    labelDS = arrayDatastore(dataLabels);
end
