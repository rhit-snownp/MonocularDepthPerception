clc
clear all
close all

%Test Training of Pipeline 1 on NYU Depth Dataset


%Load in the NYU Depth Dataset
relativePath = "nyu_depth_v2_labeled.mat";
[inputDataDepths,inputDataImages] = ReadNYUDepthToDatastore(relativePath);



%Load in the defined Pipeline 1 Network Architecture

 [layers] = pipeline1();
 analyzeNetwork(layers)

