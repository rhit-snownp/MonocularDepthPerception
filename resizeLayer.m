classdef resizeLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        inputSize
        outputSize
    end
    properties (Learnable)
    end
    methods
        function layer = resizeLayer(name)
            layer.Name = name;
           
        end
        function [Z] = predict(layer, X)
            Z = dlresize(X, "OutputSize", [304 228]);
        end
    end
    
end