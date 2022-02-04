clc; close all; clear variables; 
%%
testCombined = ReadTestData("images\test");
%%
images = testCombined.read();
inputImg = images{1};
% inputImg = ones([227 227 3]);
out = exp(predict(net, inputImg));
scaledOutput = uint8(interp1([min(min(out)), max(max(out))], [0,255],out));
imshow(imtile({imresize(scaledOutput,6),imresize(inputImg,6)}));


function [testCombined] = ReadTestData(relativePath)
    inputDataImages = imageDatastore(relativePath,"ReadFcn", @loadImage,"IncludeSubfolders",true);
%     augDataImages = augmentedImageDatastore([304, 228], inputDataImages);
    inputDataDepths = imageDatastore(relativePath, 'ReadFcn',@loadDIODEZDepth,'FileExtensions','.npy',"IncludeSubfolders",true);
%     augDataDepths = augmentedImageDatastore([76,57], inputDataDepths);

    testCombined = combine(inputDataImages, inputDataDepths);

    function data = loadDIODEZDepth(filename)
        addpath npy-matlab\
        data = readNPY(filename);
        data = imresize(data,[76,57]);
    end

    function data = loadImage(filename)
        im = imread(filename);
        im = imgaussfilt(im,2);
        data = imresize(im, [304 228]);
    end
end