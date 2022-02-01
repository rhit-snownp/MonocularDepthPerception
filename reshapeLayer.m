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
%             layer.Description = "reshape layer";
%             if length(outputSize) ~= 2
%                 ME = MException('reshapeLayer:OutputError', 'Output must be 2 dimensional');
%                 throw(ME)
%             elseif inputSize ~= outputSize(1)*outputSize(2)
%                 ME = MException('reshapeLayer:OutputError', 'Input and output size do not match');
%                 throw(ME)
%             end
%             layer.inputSize = inputSize;
%             layer.outputSize = outputSize;
        end

        function [Z] = predict(layer, X)
            Z = reshape(X, 76, 57, 1, []);
        end
    end
    
end