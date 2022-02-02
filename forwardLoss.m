function loss = forwardLoss(Y, T)
            % loss = forwardLoss(layer, Y, T) returns the SIE loss between
            % the predictions Y and the training targets T.

            % T has dimensions height x width x channels (1) x N 
            % N is minibatch size
            
            T = T + 1e-7; %dont get Nan
            Y = Y + 1e-7;
            nPixels = size(T,1) * size(T,2);
            alpha = ones(size( T(:,:,1,:) )).*(sum(log(abs(T)) - log(abs(Y)),[1,2])/nPixels);
            sumLoss = sum((log(Y) - log(T) + alpha).^2,'all');
    
            % Take mean over mini-batch.
            N = size(Y,4);
            loss = sumLoss/N;
end
% 
% al =
% 
%          0   -0.4055         0
%    -2.3026    0.4700         0
%     1.6094   -1.0986   -0.5878