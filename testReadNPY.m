addpath npy-matlab\
a = readNPY("train/indoors/scene_00000/scan_00001/00000_00001_indoors_060_000_depth.npy");
a = normalize(a);
imshow(a)

