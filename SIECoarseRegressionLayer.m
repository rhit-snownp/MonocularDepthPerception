classdef SIECoarseRegressionLayer < nnet.layer.RegressionLayer
    % Scale-Invariant Error
    
    methods
        function layer = SIECoarseRegressionLayer(name)
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
            T = imresize(T,[74,55,1]);
            nPixels = sum(T(:,:,1,1),'all');
            alpha = ones(size( T(:,:,1,:) )).*(sum(log(T(:,:,1,:)) - log(Y),[1,2])/nPixels);
            sumLoss = sum((log(Y) - log(T(:,:,1,:)) + alpha).^2,'all');
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = sumLoss/N;
        end
    end
end