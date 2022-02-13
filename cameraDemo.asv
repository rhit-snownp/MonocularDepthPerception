clear all

load("combinedNet2.mat")

cam = webcam(1);
%%

img = snapshot(cam);

img = imresize(img, [304 228]);

depth = net.predict(img);

subplot(1,2,1)
imshow(img);
subplot(1,2,2)
imagesc(depth);title("output depthmap");colorbar;axis equal;