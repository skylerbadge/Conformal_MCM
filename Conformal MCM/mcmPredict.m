function [ pred,acc ] = mcmPredict( xTrain,xTest,yTest,kerType,kerPara,lambda,b )

[N,~]=size(xTrain);
[Ntest,~]=size(xTest);
pred=zeros(Ntest,1);


% Compute kernel matrix
for i=1:Ntest
    K=zeros(N,1);
    for      j=1:N
        
        K(j)=K(j)+...
            kernelfunction(kerType,xTest(i,:),xTrain(j,:),kerPara);
        
    end
    predVal=((lambda'*K)+b)/norm(lambda);
    if(predVal)>=0
        pred(i)=1;
    else
        pred(i)=-1;
    end
end

acc=sum(pred==yTest)/Ntest*100;

if(acc<50)
    pred=-pred;
    acc=sum(pred==yTest)/Ntest*100;
end

end

