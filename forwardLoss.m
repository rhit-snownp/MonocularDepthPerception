function loss = forwardLoss(Y, T)
            % loss = forwardLoss(layer, Y, T) returns the SIE loss between
            % the predictions Y and the training targets T.

            % T has dimensions height x width x channels (1) x N 
            % N is minibatch size
            batchSize = size(T,4);
            idx = T>0;
            for i=1:batchSize
                di = Yi(idx) - log(abs(Ti(idx)));
                n = length(di);
                di2 = di.^2;
                first_term = sum(di2)/n;
                second_term = 0.5*(sum(di).^2)/n^2;
                loss_vec(i) = first_term - second_term;
                loss = mean(loss_vec);
            end
end
