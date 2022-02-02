classdef SIERegressionLayer < nnet.layer.RegressionLayer
    % Scale-Invariant Error
    
    methods
        function layer = SIERegressionLayer(name)
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Scale-Invariant Error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            % loss = forwardLoss(layer, Y, T) returns the SIE loss between
            % the predictions Y and the training targets T.

            % T has dimensions height x width x channels (1) x N 
            % N is minibatch size
            
            T = T + 1e-7; %dont get Nan
            Y = Y + 1e-7;
            nPixels = size(T,1) * size(T,2);
            alpha = ones(size( T(:,:,1,:) )).*(sum(log(abs(T)) - log(abs(Y)),[1,2])/nPixels);
            sumLoss = sum((log(abs(Y)) - log(abs(T)) + alpha).^2,'all');
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = sumLoss/N;
        end
    end
end