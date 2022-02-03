addpath npy-matlab\
% imread("images/test/indoors/scene_00019/scan_00183/00019_00183_indoors_000_010_depth.npy")
a = readNPY("images/test/indoors/scene_00019/scan_00183/00019_00183_indoors_000_010_depth.npy");
% a = normalize(a);
imshow(a)

