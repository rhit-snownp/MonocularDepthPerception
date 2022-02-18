%Extract NYU Dataset into Pairs
clc;
clear all
close all;

load("nyu_depth_v2_labeled.mat");

%% Script to Extract NYU Depth Dataset V2 into PNG Images and PNG Depth Maps

tic

for currentImageIndex = 1:length(scenes)

%Extract a depth image, and the file name and save it
currentDepthImage = depths(:,:,currentImageIndex);
depthImageFilename = char(rawDepthFilenames(currentImageIndex));

%Extract an image, and the file name and save it
currentImage = images(:,:,:,currentImageIndex);
imageFilename = char(rawRgbFilenames(currentImageIndex));

imwrite(currentDepthImage,sprintf("Depth %03d.png",currentImageIndex),'png');
imwrite(currentImage,sprintf("Image %03d.png",currentImageIndex),'png');

end

toc