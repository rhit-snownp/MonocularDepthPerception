classdef SIERegressionLayer < nnet.layer.RegressionLayer
    % Scale-Invariant Error loss function
    
    methods
        function layer = SIERegressionLayer(name)
            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = 'Scale-Invariant Error';
        end
        
        function loss = forwardLoss(layer, Y, T)
            batchSize = size(T,4);
            di = log(abs(T)) - Y;
            di(T==0) = 0; %if the depth is invalid in ground truth, set it to zero in di

            % n counts the number of valid depth readings in each image
            % within the minibatch. This way we are dividing by the right
            % value in the equation
            n = size(T,1)*size(T,2)*ones(1,1,1,batchSize); %initialize vector (in 4th dim) of n (#pixels in depth) repeated
            for i=1:batchSize
                n(1,:,1,i) = n(1,:,1,i) - sum(di(:,:,1,i)==0,"all"); % reduce n by the number of invalid pixels in each image
            end

            %this is all directly from the eigen paper
            di2 = di.^2;
            first_term = sum(di2,[1,2,3])./n;
            second_term = 0.5*(sum(di,[1,2,3]).^2)./(n.^2);
            loss = mean(first_term - second_term);
        end
    end
end