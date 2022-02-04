function loss = forwardLoss(Y, T)
            batchSize = size(T,4);
%             di = Y - log(abs(T));
            di = log(abs(T)) - Y;
            di(T==0) = 0;
            n = size(T,1)*size(T,2)*ones(1,1,1,batchSize);
            for i=1:batchSize
                n(1,:,1,i) = n(1,:,1,i) - sum(di(:,:,1,i)==0,"all");
            end
            di2 = di.^2;
            first_term = sum(di2,[1,2,3])./n;
            second_term = 0.5*(sum(di,[1,2,3]).^2)./(n.^2);
            loss = mean(first_term - second_term);
        end