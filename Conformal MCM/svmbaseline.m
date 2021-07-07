clc;
clear all;

result=[];

for dataset=[1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29,30]%change the number from 1 to 30 you can put this in a loop like this
    % for dataset=1:30
   
    filename= sprintf('%d.mat',dataset);
    folds=sprintf('%dfold.mat',dataset);
    load(strcat('data_folds/',filename));
    load(strcat('data_folds/',folds));
    clc;
    disp(dataset);
    
    X=x;
    Y=y;
    nfolds=5;
    s=size(X,1);%size of training data  
    %hyperparameters to be initialized here
    gamma=[1e-03,1e-02,1e-01,1,10];
    cParams = [1e-03,1e-02,1e-01,1,10];

    %similarly you can define beta here for RBF kernel and C for cost
    
    t1=[];
    t2=[];
    t3=[];
    t4=[];
    t5=[];    
    
    bestKegam=0;
    bestAcc=0;
    Cbest = 0 ;
    tic;
%     for i=[1,2,3]
    for i=1:nfolds
        i
        xTrain0=[];
        yTrain=[];
        xTest0=[];
        yTest=[];

        test = (indices == i);
        train = ~test;
        for j=1:s
            if(train(j)==1)
                xTrain0=[xTrain0;X(j,:)];
                yTrain=[yTrain;Y(j,:)];
            end
            if(test(j)==1)
                xTest0=[xTest0;X(j,:)];
                yTest=[yTest;Y(j,:)];
            end
        end
        %data preprocessing
        me=mean(xTrain0);
        std_dev=std(xTrain0);
        xTrain = zeros(size(xTrain0));
        xTest = zeros(size(xTest0));
        for n=1:size(xTrain,2)
            if(std_dev(n)~=0)
                xTrain(:,n)=(xTrain0(:,n)-me(n))./std_dev(n);
            else
                xTrain(:,n)=(xTrain0(:,n)-me(n));
            end
        end
        for n=1:size(xTest,2)
            if(std_dev(n)~=0)
                xTest(:,n)=(xTest0(:,n)-me(n))./std_dev(n);
            else
                xTest(:,n)=(xTest0(:,n)-me(n));
            end
        end
        AccFold=0;
        trainAccFold=0;
        for Ctest=cParams
            for gam=gamma
                try
                    model = svmtrain(yTrain, xTrain, sprintf('-t 3 -g %d -c %d',gam,Ctest));
                    [~, testAcc,~] = svmpredict(yTest, xTest, model);
                    [~, trainAcc,~] = svmpredict(yTrain,xTrain, model);
                    testAcc=testAcc(1);
                    trainAcc=trainAcc(1);
                    if(testAcc >= AccFold)
                        AccFold = testAcc;
                        trainAccFold = trainAcc;
                        nsv=model.totalSV;
                    end

                if(testAcc >= bestAcc)
                    bestAcc = testAcc;
                    besttrainAcc = trainAcc;
                    Cbest = Ctest ;
                    bestgam = gam ;
                end
                 catch
                    testAcc=0;
                    trainAcc=0;
                end
            end
        end
         
        t1=[t1;trainAccFold];
        t2=[t2;AccFold];
        t3=[t2;nsv];
         

    end
    avg1=mean(t1);
    std1=std(t1);
    avg2=mean(t2);
    std2=std(t2);
    avg3=mean(t3);
    std3=std(t3);
    
%   r=[avg1 avg2 avg3 std1 std2 std3 C1(a1) C2(a2) d_min(a3)];
    
    r=[dataset avg1 std1 avg2 std2 avg3 std3 Cbest bestgam ];
    result=[result;r];
    
    fprintf(2,'MCM : Avg Accuracy :  %.3f     C: %.3f    P: %.3f \n',avg2,Cbest,bestgam);
    xlswrite(strcat(int2str(dataset),'_svm_baseline_gs.xlsx'),result)
end
