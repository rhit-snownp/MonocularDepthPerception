function [inputDataDepths,inputDataImages] = ReadDIODEToDatastore(relativePath)

    inputDataDepths = imageDatastore(relativePath, 'ReadFcn',@loadDIODEZDepth,'FileExtensions','.npy',"IncludeSubfolders",true);
    inputDataImages = imageDatastore(relativePath,"IncludeSubfolders",true);
    
    
    function data = loadDIODEZDepth(filename)
        addpath npy-matlab\
        data = readNPY(filename);
        data = repmat(data, [1,1,3]);
%         data = imresize(data,[304 228]);
    end

end