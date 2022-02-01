function loss = forwardLoss(Y, T)
            % loss = forwardLoss(layer, Y, T) returns the SIE loss between
            % the predictions Y and the training targets T.

            % T has dimensions height x width x channels (1) x N 
            % N is minibatch size
            
            nPixels = sum(T(:,:,1,1),'all');
            alpha = ones(size(T, [1,2])).*(sum(log(abs(T)) - log(abs(Y)), [1,2])/nPixels);
            sumLoss = sum((log(abs(Y)) - log(abs(T(:,:,1,:))) + alpha).^2,'all');
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = sumLoss/N;
        end