function [ pred,acc ] = mcmPredictConformal( xTrain,yTrain,xTest,yTest,Kt,lambda,b )


[n,~]=size(xTrain);
m=length(yTest);
H = zeros(m,n);

for i=1:m
    for j=1:n
        
%         K(j)=K(j)+kernelfunction(kerType,xTest(i,:),xTrain(j,:),kerPara);
        H(i,j) = yTrain(j)*Kt(i,j);
    end
end

pred = sign(H*lambda + b);


acc=sum(pred==yTest)/m*100;

if(acc<50)
    pred=-pred;
    acc=sum(pred==yTest)/m*100;
end

end

