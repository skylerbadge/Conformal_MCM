function [ Cbest,  bestKerPara] = tuneMCM( xTrain, yTrain, kerTypeMCM , cParams , kerParams )
    
    [N, D]=size(xTrain);
    % 80:20 split for validation
    split_pt=round(0.8*N);
    xTrain1=xTrain(1:split_pt,:);
    yTrain1=yTrain(1:split_pt,:);
    xValid=xTrain(split_pt+1:end,:);
    yValid=yTrain(split_pt+1:end,:);
    bestAcc=0;
    Cbest=0;bestKerPara=0;
    
    % Grid Search over the C and gamma parameters
    for i=1:length(cParams)
    
        for k=1:length(kerParams)
            C=cParams(i);
            kerParaMCM=kerParams(k);
            try
                [ lambda,b,h ] = mcm_linear_efs( xTrain1, yTrain1, kerTypeMCM, kerParaMCM, C );
            catch
                lambda = rand(size(xTrain1,1),1);
                b = rand;
            end
            [ pred , acc_mcm ] = mcmPredict( xTrain1,xValid,yValid,kerTypeMCM,kerParaMCM,lambda,b );
            acc_mcm
            kerParaMCM
            C
            
            if(acc_mcm >= bestAcc)
                bestAcc = acc_mcm;
                Cbest=C;
                bestKerPara=kerParaMCM;
            end 
        end
    end
    
    fprintf(2,'Tuning done: C=%.2f, kerParaMCM=%.2f, bestTrainAcc=%.2f \n', Cbest, bestKerPara, bestAcc);
    

end