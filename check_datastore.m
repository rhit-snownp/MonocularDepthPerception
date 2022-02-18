%Utility file to view images in datastore
clc; close all; clear variables; 

[trainCombined, valCombined] = ReadDIODEToDatastore("images\train\indoors\");

while(true)
    out = valCombined.read(); subplot(1,2,1); imshow(out{1}); subplot(1,2,2); imshow(uint8(interp1([min(min(out{2})), max(max(out{2}))], [0,255],out{2})) ,'Colormap', hsv(256))
    input("enter")
    for i=1:500
        out = valCombined.read();
    end
end