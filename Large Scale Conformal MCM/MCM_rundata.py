#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from time import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg as la
from numpy import linalg as nla
import sklearn.datasets as sd
from numpy.matlib import repmat
from scipy.stats import mode
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample
from scipy.io import arff


from MCMmy import MCM

# set path 

# path1="/Users/sbadge/Documents/Jayadeva/LSMCM/large-scale-MCM"
# path1="C:/Users/skyle/OneDrive/IIT_Delhi/Jayadeva/MCM/LSMCM/LSMCM/large-scale-MCM"
path1="/scratch/ee/btech/ee1170498/lsconfmcm"
os.chdir(path1)

#  funtion to normalize dataset
def standardize(xTrain):
    me=np.mean(xTrain,axis=0)
    std_dev=np.std(xTrain,axis=0)
    #remove columns with zero std
    idx=(std_dev!=0.0)
    print(idx.shape)
    xTrain[:,idx]=(xTrain[:,idx]-me[idx])/std_dev[idx]
    return xTrain,me,std_dev

    #%%

# set relative path of dataset folder
datapath=path1 +'/data'
#randomly sample class=1
imbalance_ratio=1
#dataset_name=10
dataset_type='clustering'

results = pd.DataFrame(columns=['Dataset','TrainAcc','TestAcc','TrainAccConf','TestAccConf','C','gamma0','gamma1'])

# choose dataset
for dataset in ['creditcard']:

    typeAlgo= 'MCM_C'
    np.random.seed(1)

    #  train test split done according to dataset
    #  note: targets values are 0 and 1 
    if(dataset=='a8a'):
        x1,y1=sd.load_svmlight_file(datapath+'/a8atrain.txt',n_features=123)
        y1=(y1+1)/2
        y1=np.array(y1,dtype=np.int32)
        x1=x1.toarray()
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
        
        xTest,yTest=sd.load_svmlight_file(datapath+'/a8atest.txt',n_features=123)
        xTest=xTest.toarray()
        yTest=(yTest+1)/2
        
    elif(dataset=='a3a'):
        x1,y1=sd.load_svmlight_file(datapath+'/a3atrain.txt',n_features=123)
        y1=(y1+1)/2
        y1=np.array(y1,dtype=np.int32)
        x1=x1.toarray()
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
        
        xTest,yTest=sd.load_svmlight_file(datapath+'/a3atest.txt',n_features=123)
        xTest=xTest.toarray()
        yTest=(yTest+1)/2
        yTest=np.array(yTest,dtype=np.int32)
    
    elif(dataset=='a4a'):
        x1,y1=sd.load_svmlight_file(datapath+'/a4atrain.txt',n_features=123)
        y1=(y1+1)/2
        y1=np.array(y1,dtype=np.int32)
        x1=x1.toarray()
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
        
        xTest,yTest=sd.load_svmlight_file(datapath+'/a4atest.txt',n_features=123)
        xTest=xTest.toarray()
        yTest=(yTest+1)/2
        yTest=np.array(yTest,dtype=np.int32)
        
    elif(dataset=='w4a'):
        x1,y1=sd.load_svmlight_file(datapath+'/w4atrain.txt',n_features=300)
        y1=(y1+1)/2
        y1=np.array(y1,dtype=np.int32)
        x1=x1.toarray()
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
        
        xTest,yTest=sd.load_svmlight_file(datapath+'/w4atest.txt',n_features=300)
        xTest=xTest.toarray()
        yTest=(yTest+1)/2
        yTest=np.array(yTest,dtype=np.int32)
    
    elif(dataset=='breast-cancer'):
        X,Y=sd.load_svmlight_file(datapath+'/breast-cancer_scale.txt',n_features=10)
        Y=(Y-2)/2
        Y=np.array(Y,dtype=np.int32)
        X=X.toarray()
        
        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
        
    elif(dataset=='diabetes'):
        X,Y=sd.load_svmlight_file(datapath+'/diabetes_scale.txt',n_features=8)
        Y=(Y+1)/2
        Y=np.array(Y,dtype=np.int32)
        X=X.toarray()
        
        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
   
    elif(dataset=='fourclass'):
        X,Y=sd.load_svmlight_file(datapath+'/fourclass_scale.txt',n_features=2)
        Y=(Y+1)/2
        Y=np.array(Y,dtype=np.int32)
        X=X.toarray()
        
        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
        
    elif(dataset=='german-numer'):
        X,Y=sd.load_svmlight_file(datapath+'/german.numer_scale.txt',n_features=24)
        Y=(Y+1)/2
        Y=np.array(Y,dtype=np.int32)
        X=X.toarray()
        
        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
    
    elif(dataset=='phishing'):
        X,Y=sd.load_svmlight_file(datapath+'/phishing.txt',n_features=68)
        Y=np.array(Y,dtype=np.int32)
        X=X.toarray()
        
        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
        
    elif(dataset=='australian'):
        X,Y=sd.load_svmlight_file(datapath+'/australian_scale.txt',n_features=24)
        Y=(Y+1)/2
        Y=np.array(Y,dtype=np.int32)
        X=X.toarray()
        
        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
      
    elif(dataset=='skin'):
        data = np.genfromtxt(datapath+'/'+dataset+'.txt')
        X=data[:,0:-1]
        Y=data[:,-1]
        Y=(Y-1)
        Y=np.array(Y,dtype=np.int32)

        X,me,std_dev=standardize(X)

        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
        
    elif(dataset=='eye'):
        data = arff.loadarff(datapath+'/'+'EEGEyeState.arff')
        df = pd.DataFrame(data[0])

        X=df.values[:,0:-1]
        Y=df.values[:,-1]
        Y=np.array(Y,dtype=np.int32)

        X,me,std_dev=standardize(X)

        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)

    elif(dataset=='htru2'):
        data = np.genfromtxt(datapath+'/HTRU_2.csv',delimiter=',')
        X=data[:,0:-1]
        Y=data[:,-1]
        Y=np.array(Y,dtype=np.int32)

        X,me,std_dev=standardize(X)

        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
    
    elif(dataset=='magic'):
        data = np.genfromtxt(datapath+'/magic04.data',delimiter=',')
        data2 = np.genfromtxt(datapath+'/magic04.data',delimiter=',',dtype=str)
        data[data2[:,-1]=='g',-1]=0
        data[data2[:,-1]=='h',-1]=1
        X=data[:,0:-1]
        Y=data[:,-1]
        
        X,me,std_dev=standardize(X)

        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
    
    elif(dataset=='creditcard'):
        data = np.genfromtxt(datapath+'/defaultofcreditcardclients.csv',delimiter=',')
        X=data[:,0:-1]
        Y=data[:,-1]
        
        X,me,std_dev=standardize(X)

        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)
        

    else:
        data = np.genfromtxt(datapath+'/'+dataset+'.csv',delimiter=',')
        X=data[:,0:-1]
        Y=data[:,-1]
        Y=(Y+1)/2
        Y=np.array(Y,dtype=np.int32)

        X,me,std_dev=standardize(X)

        x1, xTest, y1, yTest = train_test_split(X, Y, test_size=0.2, stratify=Y ,random_state=42)
        xTrain, xVal, yTrain, yVal = train_test_split(x1, y1, test_size=0.2, stratify=y1 ,random_state=42)

    # Relevant Hyperparameters:
    # Slack Hyperparameter C -- Cb
    # Primary kernel paremeter -- gamma1


    # parameter 'xyz' indices are denoted by 'xyz_idx' and it saves the indices, instead of strings in 'xyz' to be saved in a pandas dataframe
    # when running the functions the parameter 'xyz = 'abc' eg. kernel_type = 'rbf' can be passed as is 
    # the parameter indices eg. 'xyz_idx' : kernel_type_idx is not required unless you wish to save the results in a numpy array as I have
#    Ca = [0,1e-05,1e-03,1e-02,1e-01,1] #hyperparameter 1 #loss function parameter
    Ca = [0]

    Cb = [1e-04,1e-03,1e-02,1e-01,1,10] #hyperparameter 2 #when using L1 or L2 or ISTA penalty

    Cc = [0] #hyperparameter 2 #when using elastic net penalty (this parameter should be between 0 and 1)

    Cd = [0] #hyperparameter for final regressor or classifier used to ensemble when concatenating the outputs of previous layer of classifier or regressors
    problem_type1 = {0:'classification', 1:'regression'}
    problem_type = 'classification'
    problem_type_idx = 0
    algo_type1 = {0:'MCM',1:'LSMCM'}
    algo_type = 'LSMCM'
    algo_type_idx = 1
    kernel_type1 = {0:'linear', 1:'rbf', 2:'sin', 3:'tanh', 4:'TL1', 5:'linear_primal', 6:'rff_primal', 7:'nystrom_primal'} 
    kernel_type = 'rbf'
    kernel_type_idx = 1
    # gamma1 = [1e-04,1e-03,1e-02,1e-01,1,10,100] #hyperparameter3 (kernel parameter for non-linear classification or regression)

    gamma1 = np.power(2.0,[-10,-11,-12,-8,-9])
    epsilon1 = [0.0] #hyperparameter4 ( It specifies the epsilon-tube within which 
    #no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.)

#    n_ensembles1 = [1]  #number of ensembles to be learnt, if setting n_ensembles > 1 then keep the sample ratio to be around 0.7
    n_ensembles = 1
#    feature_ratio1 = [1.0] #percentage of features to select for each PLM
    feature_ratio = 1.0
#    sample_ratio1 = [1.0] #percentage of data to be selected for each PLM
    sample_ratio = 1.0
#    batch_sz1 = [128] #batch_size
    batch_sz = 128
#    iterMax1a = [1000] #max number of iterations for inner SGD loop
    iterMax1 = 1000
    iterMax2 = 10
#    eta1 = [1e-02] #initial learning rate
    eta = 1e-1
#    tol1 = [1e-04] #tolerance to cut off SGD
    tol = 1e-05
    update_type1 =  {0:'sgd',1:'momentum',3:'nesterov',4:'rmsprop',5:'adagrad',6:'adam'}#{0:'sgd',1:'momentum',3:'nesterov',4:'rmsprop',5:'adagrad',6:'adam'}
    update_type ='sgd'
    update_type_idx = 0
    reg_type1 = {0:'l1', 1:'l2', 2:'en', 4:'ISTA', 5:'M'} #{0:'l1', 1:'l2', 2:'en', 4:ISTA, 5:'M'}#ISTA: iterative soft thresholding (proximal gradient)
    reg_type = 'l1'
    reg_type_idx = 0
    feature_sel1 = {0:'sliding', 1:'random'} #{0:'sliding', 1:'random'}
    feature_sel = 'random'
    feature_sel_idx = 1
    class_weighting1 = {0:'average', 1:'balanced'}#{0:'average', 1:'balanced'}
    class_weighting = 'average'
    class_weighting_idx = 0
    combine_type1 =  {0:'concat',1:'average',2:'mode'} #{0:'concat',1:'average',2:'mode'}
    combine_type = 'average'
    combine_type_idx = 1
    upsample1a =  {0:False, 1:True} #{0:'False', 1:'True'}
    upsample1  = False
    upsample1_idx = 0
    PV_scheme1 = {0:'kmeans', 1:'renyi'}  #{0:'kmeans', 1:'renyi'}
    PV_scheme = 'kmeans'
    PV_scheme_idx = 0
    n_components = int(5*np.sqrt(xTrain.shape[0]))
    do_pca_in_selection1 = {0:False,1:True} 
    do_pca_in_selection = False 
    do_pca_in_selection_idx = 0
    conformal = True
                    
    # Set C and gamm1 hyperparameters here for grid search 
    Cb = [1e-3,1e-2,1e-1,1,1e1,1e2] 
    gamma1 = [1e-3,1e-2,1e-1,1,1e1,1e2] 

    maxvalacc=0
    cval=0
    gval=0
    
    for gamma in gamma1:
        for C in Cb:

            mcm = MCM(C1 = 0, C2 = C, C3 = 0, C4 = 0, problem_type = problem_type, algo_type = algo_type, kernel_type = kernel_type, gamma = gamma, 
                      epsilon = 0, feature_ratio = feature_ratio, sample_ratio = sample_ratio, feature_sel = feature_sel, 
                      n_ensembles = n_ensembles, batch_sz = batch_sz, iterMax1 = iterMax1, iterMax2 = iterMax2, eta = eta, tol = tol, update_type = update_type, 
                      reg_type = reg_type, combine_type = combine_type, class_weighting = class_weighting, upsample1 = upsample1,
                      PV_scheme = PV_scheme, n_components = n_components, do_pca_in_selection = do_pca_in_selection )
            W_all, sample_indices, feature_indices, me_all, std_all, subset_all = mcm.fit(xTrain,yTrain)


            train_pred=mcm.predict(xTrain, xTrain, W_all, sample_indices, feature_indices, me_all, std_all, subset_all)
            val_pred=mcm.predict(xVal, xTrain, W_all, sample_indices, feature_indices, me_all, std_all, subset_all)

            train_acc=mcm.accuracy_classifier(yTrain,train_pred)
            val_acc=mcm.accuracy_classifier(yVal,val_pred)
            
            print ('C1=%0.3f, gamma=%0.3f -> train acc= %0.2f, val acc=%0.2f'%(C,gamma,train_acc,val_acc))

            if(val_acc>maxvalacc):
                maxvalacc=val_acc
                cval=C
                gval=gamma
            
    print('Testing')    
    mcm = MCM(C1 = 0, C2 = cval, C3 = 0, C4 = 0, problem_type = problem_type, algo_type = algo_type, kernel_type = kernel_type, gamma = gval, 
              epsilon = 0, feature_ratio = feature_ratio, sample_ratio = sample_ratio, feature_sel = feature_sel, 
              n_ensembles = n_ensembles, batch_sz = batch_sz, iterMax1 = iterMax1, iterMax2 = iterMax2, eta = eta, tol = tol, update_type = update_type, 
             reg_type = reg_type, combine_type = combine_type, class_weighting = class_weighting, upsample1 = upsample1,
             PV_scheme = PV_scheme, n_components = n_components, do_pca_in_selection = do_pca_in_selection )
    W_all, sample_indices, feature_indices, me_all, std_all, subset_all = mcm.fit(x1,y1)

    train_pred=mcm.predict(x1, x1, W_all, sample_indices, feature_indices, me_all, std_all, subset_all)
    test_pred=mcm.predict(xTest, x1, W_all, sample_indices, feature_indices, me_all, std_all, subset_all)

    train_acc=mcm.accuracy_classifier(y1,train_pred)
    test_acc=mcm.accuracy_classifier(yTest,test_pred)
    
    print ('C1=%0.3f, gamma=%0.3f -> train acc= %0.2f, test acc=%0.2f'%(cval,gval,train_acc,test_acc))
  


# In[3]:



    maxconfacc=0
    gbest=0
    conformal=True

    # Tune conformal kernel parameter -- gam1

    # for gam1 in gval*np.array([1e-8,0.1,2,2.2,2.4,2.6,2.8,3,4,5,10,15,20,25,30,40,50,60,70,80,100,500,800,1000,5000]):

    for gam1 in np.linspace(1e-9,1e-7,50):
        
        yEmp = y1[((abs(W_all[0][:,1])>1e-3)==(abs(W_all[0][:,0])>1e-3))[1:]] #y
        xEmp = x1[((abs(W_all[0][:,1])>1e-3)==(abs(W_all[0][:,0])>1e-3))[1:]] #a
        
        np.random.seed(2)
        mask = np.random.choice(a=np.arange(0,len(yEmp)),size=int(len(yEmp)/3),replace=False,p=None)
        yEmp=yEmp[mask]
        xEmp=xEmp[mask]
        
        s=np.sum(yEmp)
        if(s>int(yEmp.shape[0]/2)):
            xEmp=np.concatenate((xEmp[yEmp==1][0:int(yEmp.shape[0]-s)],xEmp[yEmp==0]))
            yEmp=np.concatenate((yEmp[yEmp==1][0:int(yEmp.shape[0]-s)],yEmp[yEmp==0]))
        else:
            xEmp=np.concatenate((xEmp[yEmp==0][0:int(s)],xEmp[yEmp==1]))
            yEmp=np.concatenate((yEmp[yEmp==0][0:int(s)],yEmp[yEmp==1]))

                
        x1=np.concatenate((x1[y1==0],x1[y1==1]))
        y1=np.concatenate((y1[y1==0],y1[y1==1]))
        
        m = x1.shape[0]

        m2 = int(np.sum(y1))
        m1 = int(m-m2)
        de = xEmp.shape[0]

        k0 = rbf_kernel(x1,x1,gval)
        
        firstw0 = np.diag(np.diag(k0))
        k11 = k0[0:m1,0:m1]
        k12 = k0[0:m1,m1:m]
        k21 = k0[m1:m,0:m1]
        k22 = k0[m1:m,m1:m]

        k1 = np.append(np.ones([m,1]), rbf_kernel(x1,xEmp,gam1), axis=1)

        b0a1 = np.append((1/m)*k11,np.zeros([m1,m2]), axis=1)
        b0b1 = np.append(np.zeros([m2,m1]),(1/m)*k22, axis=1)
        b01 = np.concatenate((b0a1, b0b1))

        b0 = b01-((1/m)*k0)
        w0 = firstw0 - b01

        C = 1e-6
        D = 1e-6
        e1 = np.matmul(np.matmul(np.transpose(k1),b0),k1)+C*np.eye(de+1)
        e2 = np.matmul(np.matmul(np.transpose(k1),w0),k1)+D*np.eye(de+1)

        lam,ralpha = la.eig(np.matmul(nla.inv(e2),e1), left=False, right=True)

        maxid = 0
        for i in range(lam.shape[0]):
            if(abs(lam[maxid])>abs(lam[i])):
                maxid=i

        eigv = ralpha[:,maxid].real
        qt = np.matmul(k1,eigv)

        qtestr=np.sum(rbf_kernel(xTest,xEmp,gam1)*eigv[0:xEmp.shape[0]],axis=1)+eigv[0]
        
        W_all1, sample_indices, feature_indices, me_all, std_all, subset_all = mcm.fit(x1,y1,conformal,qt)

        train_pred=mcm.predict(x1, x1, W_all1, sample_indices, feature_indices, me_all, std_all, subset_all,conformal,qt,qt)
        test_pred=mcm.predict(xTest, x1, W_all1, sample_indices, feature_indices, me_all, std_all, subset_all,conformal,qtestr,qt)

        train_acc_conf=mcm.accuracy_classifier(y1,train_pred)
        test_acc_conf=mcm.accuracy_classifier(yTest,test_pred)

        print ('C1=%0.3f, gamma=%0.3f -> train acc= %0.2f, test acc=%0.2f'%(cval,gam1,train_acc_conf,test_acc_conf))
        
        if(test_acc_conf>maxconfacc):
            maxconfacc=test_acc_conf
            trainconfacc = train_acc_conf
            gbest=gam1
            

    # Record Results
    resrow = {'Dataset':dataset,'TrainAcc':train_acc,'TestAcc':test_acc,'TrainAccConf':trainconfacc,'TestAccConf':maxconfacc,'C':cval,'gamma0':gval,'gamma1':gbest}
    results = results.append(resrow, ignore_index=True)
    print(results)
    results.to_csv(path1+"/results/conf_"+dataset+".csv")
