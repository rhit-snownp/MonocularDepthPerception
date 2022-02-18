% used in the corase network to reshape the output of the last fully
% connected layer
classdef reshapeLayer < nnet.layer.Layer & nnet.layer.Formattable
    properties
        inputSize
        outputSize
    end
    properties (Learnable)
    end
    methods
        function layer = reshapeLayer(name)
            layer.Name = name;
        end

        function [Z] = predict(layer, X)
            Z = reshape(X, 76, 57, 1, []);
        end
    end
    
end