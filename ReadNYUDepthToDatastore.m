function [inputDataDepths,inputDataImages] = ReadNYUDepthToDatastore(relativePath)

%Load the .mat file, and create testing and training datasets for the
%system

inputDataDepths=fileDatastore((relativePath),'ReadFcn',@loadNYCDepth,'FileExtensions','.mat',"ReadMode","partialfile");
inputDataImages=fileDatastore((relativePath),'ReadFcn',@loadNYCImage,'FileExtensions','.mat',"ReadMode","partialfile");


function [data,variables,done] = loadNYCDepth(filename,variables)
if(isempty(variables))
variables = 1;
end
exampleObject = matfile(filename);
data = exampleObject.depths(:,:,variables);
variables = variables + 1;
done = (variables==1449);
end

function [data,variables,done] = loadNYCImage(filename,variables)
if(isempty(variables))
variables = 1;
end
exampleObject = matfile(filename);
data = exampleObject.depths(:,:,:,variables);
variables = variables + 1;
done = (variables==1449);
end
end



