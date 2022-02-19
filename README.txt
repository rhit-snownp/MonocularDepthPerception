# Monocular Depth Perception Repo

This repo contains the code for the Monocular Depth Perception Team for CSSE-464

Requirements:
MATLAB 2021b
Image Processing Toolbox
Deep Learning Toolbox
Statistics and Machine Learning Toolbox
GPU Coder
Embedded Coder
Deep Learning HDL Toolbox
Parallel Computing Toolbox
Deep Learning Toolbox Model for AlexNet Network
MATLAB Support Package for USB Webcams


`testOutput.m` is a MATLAB script that benchmarks an existing neural network with delta values. It contains detailed instructions on how to run the code.

`cameraDemo.m` is a MATLAB script that takes an existing neural network model

` removeMasks.py` is a Python Script that goes through the folder structure of the DIODE dataset and removes unneeded masks. Uses Python 3.8.1

`NYU_Dataset_Test.m` is a MATLAB script that trains a basic neural network on the NYU Dataset. Currently deprecated due to loss function

`AlexNetTransferLearning.m` is a MATLAB script that trains a heavily modified version of AlexNet for depth estimation purposes. It contains detailed instructions on how to run the code.
	
`trainCoraseNetowrk.m`is a MATLAB script that trains an Eigen Coarse network independently. It contains detailed instructions on how to run the code.

`TrainFineNetwork.m`is a MATLAB script that trains an Eigen Fine network independently. It contains detailed instructions on how to run the code.

`TrainFineViaCoarse.m`is a MATLAB script that trains the full Eigen network from existing Fine and Coarse networks. It contains detailed instructions on how to run the code.






The required folder structure is as follows:

(The train and test folders of the images are to be filled with the corresponding images from the DIODE Depth Dataset
As a preprocessing step, "removeMasks.py" should be run to remove the extraneous masks in the folders)
(The NYU Depth Dataset.mat file can be downloaded, and ExtractNYUDataset.m can be run to extract .png files from the .mat file
to be organized into the correct directories)


Monocular Depth Perception
|   .gitignore
|   calculate_threshold_metric.m
|   cameraDemo.m
|   check_datastore.m
|   combineCoarseFine.m
|   DIODE_Dataset_Test.m
|   forwardLoss.m
|   freezeWeights.m
|   NYU_Dataset_Test.m
|   pipeline1.m
|   pipeline2.m
|   ReadDIODEforCombined.m
|   ReadDIODEforFineTraining.m
|   ReadDIODEToDatastore.m
|   README.txt
|   removeMasks.py
|   reshapeLayer.m
|   resizeLayer.m
|   SIECoarseRegressionLayer.m
|   SIERegressionLayer.m
|   testOutput.m
|   testReadNPY.m
|   trainCoraseNetowrk.m
|   TrainFineNetwork.m
|   TrainFineViaCoarse.m
|
|
+---Trained Networks
|       Coarse Network 1.mat
|       coarseNet2.mat
|       Combined Network 1.mat
|       Combined Network 2 - Fine Training On Coarse.mat
|       combinedNet.mat
|       Fine Network 1.mat
|       Fine Network 2.mat
|       Fine Network 3.mat
|       Fine Network 4.mat
|       Fine Network 5.mat
|       TrainedNetwork1.mat
|       Transfer Learning Via Alexnet 1.mat
|       Transfer Learning Via Alexnet 2.mat
|       
|
+---images
|   |   testReadNPY.m
|   |   
|   +---npy-matlab
|   |       constructNPYheader.m
|   |       datToNPY.m
|   |       readNPY.m
|   |       readNPYheader.m
|   |       writeNPY.m
|   +---test
|   |   \---indoors
|   \---train
|       \---indoors
|
|
+---NYU Dataset
|   |   ExtractNYUDataset.m
|   |   nyu_depth_v2_labeled.mat
|   |   ReadNYUDepthToDatastore.m
|   |   
|   +---Testing Data
|   |   +---Input
|   |   \---Output
|   \---Training Data
|       +---Input
|       \---Output
|
|
\---Transfer Learning Attempts
        AlexNetTransferLearning.m



