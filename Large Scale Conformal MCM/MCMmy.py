# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 11:24:29 2018

@author: mayank
"""

import numpy as np
from numpy.linalg import eigh
from numpy.matlib import repmat
from scipy.stats import mode
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import IncrementalPCA
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.metrics.pairwise import (cosine_similarity, laplacian_kernel,
                                      linear_kernel, manhattan_distances,
                                      paired_euclidean_distances,
                                      pairwise_distances, polynomial_kernel,
                                      rbf_kernel, sigmoid_kernel)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import resample

#%%
class MCM:
    def __init__(self, C1 = 1.0, C2 = 1e-05, C3 =1.0, C4 =1.0, problem_type ='classification', algo_type ='MCM' ,kernel_type = 'rbf', gamma = 1e-05, epsilon = 0.1, 
                 feature_ratio = 1.0, sample_ratio = 1.0, feature_sel = 'random', n_ensembles = 1,
                 batch_sz = 128, iterMax1 = 1000, iterMax2 = 1, eta = 0.01, tol = 1e-08, update_type = 'adam', 
                 reg_type = 'l1', combine_type = 'concat', class_weighting = 'balanced', upsample1 = False,
                 PV_scheme = 'kmeans', n_components = 100, do_pca_in_selection = False ):
        self.C1 = C1 #hyperparameter 1 #loss function parameter
        self.C2 = C2 #hyperparameter 2 #when using L1 or L2 or ISTA penalty
        self.C3 = C3 #hyperparameter 2 #when using elastic net penalty (this parameter should be between 0 and 1) or margin penalty value need not be between 0 and 1
        self.C4 = C4 #hyperparameter for final regressor or classifier used to ensemble when concatenating 
#        the outputs of previos layer of classifier or regressors
        self.problem_type = problem_type #{0:'classification', 1:'regression'}
        self.algo_type = algo_type #{0:MCM,1:'LSMCM'}
        self.kernel_type = kernel_type #{0:'linear', 1:'rbf', 2:'sin', 3:'tanh', 4:'TL1', 5:'linear_primal', 6:'rff_primal', 7:'nystrom_primal'}
        self.gamma = gamma #hyperparameter3 (kernel parameter for non-linear classification or regression)
        self.epsilon = epsilon #hyperparameter4 ( It specifies the epsilon-tube within which 
        #no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.)
        self.n_ensembles = n_ensembles  #number of ensembles to be learnt, if setting n_ensembles > 1 then keep the sample ratio to be around 0.7
        self.feature_ratio = feature_ratio #percentage of features to select for each PLM
        self.sample_ratio = sample_ratio #percentage of data to be selected for each PLM
        self.batch_sz = batch_sz #batch_size
        self.iterMax1 = iterMax1 #max number of iterations for inner SGD loop
        self.iterMax2 = iterMax2 #max number of iterations for outer SGD loop
        self.eta = eta #initial learning rate
        self.tol = tol #tolerance to cut off SGD
        self.update_type = update_type #{0:'sgd',1:'momentum',3:'nesterov',4:'rmsprop',5:'adagrad',6:'adam'}
        self.reg_type = reg_type #{0:'l1', 1:'l2', 2:'en', 4:'ISTA', 5:'M'}#ISTA: iterative soft thresholding (proximal gradient), M: margin + l1
        self.feature_sel = feature_sel #{0:'sliding', 1:'random'}
        self.class_weighting = class_weighting #{0:'average', 1:'balanced'}
        self.combine_type = combine_type #{0:'concat',1:'average',2:'mode'}
        self.upsample1 = upsample1 #{0:False, 1:True}
        self.PV_scheme = PV_scheme # {0:'kmeans',1:'renyi'}
        self.n_components = n_components #number of components to choose as Prototype Vector set, or the number of features to form for kernel_approximation as in RFF and Nystroem 
        self.do_pca_in_selection = do_pca_in_selection #{0:False, 1:True}
        
    def add_bias(self,xTrain):
        N = xTrain.shape[0]
        if(xTrain.size!=0):
            xTrain=np.hstack((xTrain,np.ones((N,1))))
        return xTrain
    
    def standardize(self,xTrain):
        me=np.mean(xTrain,axis=0)
        std_dev=np.std(xTrain,axis=0)
        #remove columns with zero std
        idx=(std_dev!=0.0)
#        print(idx.shape)
        xTrain[:,idx]=(xTrain[:,idx]-me[idx])/std_dev[idx]
        return xTrain,me,std_dev
    
   
    def subset_selection(self,X,Y):
        n_components = self.n_components
        PV_scheme = self.PV_scheme
        problem_type = self.problem_type
        N = X.shape[0]
#        M = X.shape[1]
        numClasses = np.unique(Y).size
        
        use_global_sig = False
        use_global_sig1 = False
            
        all_samples = np.arange(N)
        subset=[]
        subset_per_class = np.zeros((numClasses,))
        class_dist = np.zeros((numClasses,))
        for i in range(numClasses):
            class_dist[i] = np.sum(Y == i)
            subset_per_class[i] = int(np.ceil((class_dist[i]/N)*n_components))
            
        for i in range(numClasses):
            xTrain = X[Y == i,]
            samples_in_class = all_samples[Y == i]
            N1 = xTrain.shape[0]
#                sig = np.power((np.std(xTrain)*(np.power(N1,(-1/(M+4))))),2)
            
            subset1 = list(np.arange(N1))
            temp=list(samples_in_class[subset1])
            subset.extend(temp)
                
        return subset
    
    def divide_into_batches_stratified(self,yTrain):
        batch_sz=self.batch_sz
        #data should be of the form samples X features
        N=yTrain.shape[0]    
        num_batches=int(np.ceil(N/batch_sz))
        sample_weights=list()
        numClasses=np.unique(yTrain).size
        idx_batches=list()
    
        skf=StratifiedKFold(n_splits=num_batches, random_state=1, shuffle=True)
        j=0
        for train_index, test_index in skf.split(np.zeros(N), yTrain):
            idx_batches.append(test_index)
            class_weights=np.zeros((numClasses,))
            sample_weights1=np.zeros((test_index.shape[0],))
            temp=yTrain[test_index,]
            for i in range(numClasses):
                idx1=(temp==i)
                class_weights[i]=1.0/(np.sum(idx1)+1e-09)#/idx.shape[0]
                sample_weights1[idx1]=class_weights[i]            
            sample_weights.append(sample_weights1)

            j+=1
        return idx_batches,sample_weights,num_batches
    def kernel_transform(self, X1, X2 = None, kernel_type = 'linear_primal', n_components = 100, gamma = 1.0):
        """
        X1: n_samples1 X M
        X2: n_samples2 X M
        X: n_samples1 X n_samples2 : if kernel_type is non primal
        X: n_samples1 X n_components : if kernel_type is primal
        """
        if(kernel_type == 'linear'):
            X = linear_kernel(X1,X2)
        elif(kernel_type == 'rbf'):
            X = rbf_kernel(X1,X2,1/(2*gamma))   
        else:
            print('No kernel_type passed: using linear primal solver')
            X = X1
        return X
    
    def margin_kernel(self, X1, kernel_type = 'linear', gamma =1.0):
        """
        X1: n_samples1 X M
        X: n_samples1 X n_samples1 : if kernel_type is non primal
        """
        
        if(kernel_type == 'linear'):
            X = linear_kernel(X1,X1)
        elif(kernel_type == 'rbf'):
            X = rbf_kernel(X1,X1,1/(2*gamma))   
        else:
            print('no kernel_type, returning None')
            return None
        return X
    
    def matrix_decomposition(self, X):
        """
        Finds the matrices consisting of positive and negative parts of kernel matrix X
        Parameters:
        ----------
        X: n_samples X n_samples

        Returns:
        --------
        K_plus: kernel corresponding to +ve part
        K_minus: kernel corresponding to -ve part            
        """
        [D,U]=eigh(X)
        U_plus = U[:,D>0.0]
        U_minus = U[:,D<=0.0]
        D_plus = np.diag(D[D>0.0])
        D_minus = np.diag(D[D<=0.0])
        K_plus = np.dot(np.dot(U_plus,D_plus),U_plus.T)
        K_minus = -np.dot(np.dot(U_minus,D_minus),U_minus.T)
        return K_plus, K_minus
    
    def inner_opt(self, X, Y, data1, level):
        gamma = self.gamma
        kernel_type = self.kernel_type
        iterMax2 = self.iterMax2
        iterMax1 = self.iterMax1
        tol = self.tol
        algo_type = self.algo_type
        #if data1 = None implies there is no kernel computation, i.e., there is only primal solvers applicable
        if(data1 is not None):
            #i.e., reg_type is not M, then train accordingly using either l1, l2, ISTA or elastic net penalty
            if(algo_type == 'MCM'):
                W,f,iters,fvals = self.train(X, Y, level, K_plus = None, K_minus = None, W = None)
            elif(algo_type == 'LSMCM'):
                W,f,iters,fvals = self.train_LSMCM(X, Y, level, K_plus = None, K_minus = None, W = None)
            else:
                print('Wrong algo selected! Using MCM instead!')
                W,f,iters,fvals = self.train(X, Y, level, K_plus = None, K_minus = None, W = None)
            return W, f, iters, fvals                
        else:
            #i.e., data1 is None -> we are using primal solvers with either l1, l2, ISTA or elastic net penalty
            if(algo_type == 'MCM'):
                W,f,iters,fvals = self.train(X,Y,level, K_plus = None, K_minus = None, W = None)
            elif(algo_type == 'LSMCM'):
                W,f,iters,fvals = self.train_LSMCM(X,Y,level, K_plus = None, K_minus = None, W = None)
            else:
                print('Wrong algo selected! Using MCM instead!')
                W,f,iters,fvals = self.train(X,Y,level, K_plus = None, K_minus = None, W = None)

            return W,f,iters,fvals           
        
        return W,f,iters,fvals
        
    def select_(self, xTest, xTrain, kernel_type, subset, idx_features, idx_samples):
        #xTest corresponds to X1
        #xTrain corresponds to X2 
        if(kernel_type == 'linear' or kernel_type =='rbf' or kernel_type =='sin' or kernel_type =='tanh' or kernel_type =='TL1'):            
            X2 = xTrain[idx_samples,:]
            X2 = X2[:,idx_features] 
            X2 = X2[subset,]
            X1 = xTest[:,idx_features]
        else:
            X1 = xTest[:,idx_features]
            X2 = None
        return X1, X2
    
    def normalize_(self,xTrain, me, std):
        idx = (std!=0.0)
        xTrain[:,idx] = (xTrain[:,idx]-me[idx])/std[idx]
        return xTrain
    
    def fit(self,xTrain,yTrain,conformal=False,qt=None):
        #xTrain: samples Xfeatures
        #yTrain: samples
        #for classification: entries of yTrain should be between {0 to numClasses-1}
        #for regresison  : entries of yTrain should be real values
        N = xTrain.shape[0] #samples
        M = xTrain.shape[1]
        if(self.problem_type =='classification'):
            numClasses=np.unique(yTrain).size
        
        feature_indices=np.zeros((self.n_ensembles,int(M*self.feature_ratio)),dtype=np.int32)
        sample_indices=np.zeros((self.n_ensembles,int(N*self.sample_ratio)),dtype=np.int32)
        
        W_all={}
        me_all= {}
        std_all = {}
        subset_all = {}
        if(self.combine_type=='concat'):    
            P_all=np.zeros((N,self.n_ensembles*numClasses)) #to concatenate the classes
            
        level=0            
        gamma = self.gamma
        kernel_type = self.kernel_type
        n_components = self.n_components
        for i in range(self.n_ensembles):
            print('training PLM %d'%i)
            
            if(self.sample_ratio!=1.0):
                idx_samples=resample(np.arange(0,N), n_samples=int(N*self.sample_ratio), random_state=i,replace=False)
            else:
                idx_samples = np.arange(N)
            
            if(self.feature_ratio!=1.0):
                idx_features=resample(np.arange(0,M), n_samples=int(M*self.feature_ratio), random_state=i,replace=False)
            else:
                idx_features = np.arange(0,M)   
                
            feature_indices[i,:] = idx_features
            sample_indices[i,:] = idx_samples
            
            xTrain_temp = xTrain[idx_samples,:]
            xTrain_temp = xTrain_temp[:,idx_features] 
            
            yTrain1 = yTrain[idx_samples,]
            
            if(kernel_type == 'linear' or kernel_type =='rbf' or kernel_type =='sin' or kernel_type =='tanh' or kernel_type =='TL1'):
                subset = self.subset_selection(xTrain_temp,yTrain1)
                data1 = xTrain_temp[subset,]
                subset_all[i] = subset
            else:
                subset_all[i] = []
                data1 = None

            xTrain1 = self.kernel_transform( X1 = xTrain_temp, X2 = data1, kernel_type = kernel_type, n_components = n_components, gamma = gamma)
            
            # CONFORMAL
            if(conformal):
                xTrain1=((xTrain1*qt).T*qt).T
            #
 
            #standardize the dataset
            xTrain1, me, std  = self.standardize(xTrain1)
            me_all[i] = me
            std_all[i] = std
                
            if(self.problem_type == 'classification'):
#                W,f,iters,fvals=self.train(xTrain1,yTrain1,level)            
                W,f,iters,fvals = self.inner_opt(xTrain1, yTrain1, data1, level)
                W_all[i]=W # W will be of the shape (M+2,numClasses)

        return W_all, sample_indices, feature_indices, me_all, std_all, subset_all
        
                
    def train(self, xTrain, yTrain, level, K_plus = None, K_minus = None, W = None):
        #min D(E|w|_1 + (1-E)*0.5*|W|_2^2) + C*\sum_i\sum_(j)|f_j(i)| + \sum_i\sum_(j_\neq y_i)max(0,(1-f_y_i(i) + f_j(i)))
        #setting C = 0 gives us SVM
        # or when using margin term i.e., reg_type = 'M'
        #min D(E|w|_1) + (E)*0.5*\sum_j=1 to numClasses (w_j^T(K+ - K-)w_j) + C*\sum_i\sum_(j)|f_j(i)| + \sum_i\sum_(j_\neq y_i)max(0,(1-f_y_i(i) + f_j(i)))
        #setting C = 0 gives us SVM with margin term   
        xTrain=self.add_bias(xTrain)
        
        M=xTrain.shape[1]
        N=xTrain.shape[0]
        numClasses=np.unique(yTrain).size
        verbose = False
        if(level==0):
            C = self.C1 #for loss function of MCM
            D = self.C2 #for L1 or L2 penalty
            E = self.C3 #for elastic net penalty or margin term
        else:
            C = self.C4 #for loss function of MCM 
            D = self.C2 #for L1 or L2 penalty
            E = self.C3 #for elastic net penalty since in combining the classifiers we use a linear primal classifier
            
        iterMax1 = self.iterMax1
        eta_zero = self.eta
        class_weighting = self.class_weighting
        reg_type = self.reg_type
        update_type = self.update_type
        tol = self.tol
        np.random.seed(1)
        
        if(W is None):
            W=0.001*np.random.randn(M,numClasses)
            W=W/np.max(np.abs(W))
        else:
            W_orig = np.zeros(W.shape)
            W_orig[:] = W[:]
        
        class_weights=np.zeros((numClasses,))
        sample_weights=np.zeros((N,))
        #divide the data into K clusters
    
        for i in range(numClasses):
            idx=(yTrain==i)           
            class_weights[i]=1.0/np.sum(idx)
            sample_weights[idx]=class_weights[i]
                        
        G_clip_threshold = 100
        W_clip_threshold = 500
        eta=eta_zero
                       
        scores = xTrain.dot(W) #samples X numClasses
        N = scores.shape[0]
        correct_scores = scores[range(N),np.array(yTrain,dtype='int32')]
        mat = (scores.transpose()-correct_scores.transpose()).transpose() 
        mat = mat+1.0
        mat[range(N),np.array(yTrain,dtype='int32')] = 0.0
        thresh1 = np.zeros(mat.shape)
        thresh1[mat>0.0] = mat[mat>0.0] #for the SVM loss 
        
        f=0.0
        if(reg_type=='l2'):
            f += D*0.5*np.sum(W**2) 
        if(reg_type=='l1'):
            f += D*np.sum(np.abs(W))
        if(reg_type=='en'):
            f += D*0.5*(1-E)*np.sum(W**2)  +  D*E*np.sum(np.abs(W))
            
            
        if(class_weighting=='average'):
            f1 = C*np.sum(np.abs(scores)) + np.sum(thresh1)
            f += (1.0/N)*f1 
        else:
            f1 = C*np.sum(np.abs(scores)*sample_weights[:,None]) + np.sum(thresh1*sample_weights[:,None])
            f+= (1.0/numClasses)*f1 
        
        if(K_minus is not None):
            temp_mat = np.dot(K_minus,W_orig[0:(M-1),])
        
        
        for i in range(numClasses):
            #add the term (E/2*numclasses)*lambda^T*K_plus*lambda for margin
            if(K_plus is not None):
                w = W[0:(M-1),i]
                f2 = np.dot(np.dot(K_plus,w),w)
                f+= ((0.5*E)/(numClasses))*f2  
             #the second term in the objective function
            if(K_minus is not None):
                f3 = np.dot(temp_mat[:,i],w)
                f+= -((0.5*E)/(numClasses))*f3
        
        
        iter1=0
        print('iter1=%d, f=%0.3f'%(iter1,f))
                
        f_best=f
        fvals=np.zeros((iterMax1+1,))
        fvals[iter1]=f_best
        W_best=np.zeros(W.shape)
        iter_best=iter1
        f_prev=f_best
        rel_error=1.0
#        f_prev_10iter=f
        
        if(reg_type=='l1' or reg_type =='en' or reg_type == 'M'):
            # from paper: Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty
            u = 0.0
            q=np.zeros(W.shape)
            z=np.zeros(W.shape)
            all_zeros=np.zeros(W.shape)
        
        eta1=eta_zero 
        v=np.zeros(W.shape)
        v_prev=np.zeros(W.shape)    
        vt=np.zeros(W.shape)
        m=np.zeros(W.shape)
        vt=np.zeros(W.shape)
        
        cache=np.zeros(W.shape)
        eps=1e-08
        decay_rate=0.99
        mu1=0.9
        mu=mu1
        beta1 = 0.9
        beta2 = 0.999  
        iter_eval=10 #evaluate after every 10 iterations
        
        idx_batches, sample_weights_batch, num_batches = self.divide_into_batches_stratified(yTrain)
        while(iter1<iterMax1 and rel_error>tol):
            iter1=iter1+1            
            for batch_num in range(0,num_batches):
    #                batch_size=batch_sizes[j]
                test_idx=idx_batches[batch_num]
                data=xTrain[test_idx,]
                labels=yTrain[test_idx,] 
                N=labels.shape[0]
                scores=data.dot(W)
                correct_scores=scores[range(N),np.array(labels,dtype='int32')]#label_batches[j] for this line should be in the range [0,numClasses-1]
                mat=(scores.transpose()-correct_scores.transpose()).transpose() 
                mat=mat+1.0
                mat[range(N),np.array(labels,dtype='int32')]=0.0
                
                thresh1=np.zeros(mat.shape)
                thresh1[mat>0.0]=mat[mat>0.0]
                
                binary1 = np.zeros(thresh1.shape)
                binary1[thresh1>0.0] = 1.0                
                
                row_sum=np.sum(binary1,axis=1)
                binary1[range(N),np.array(labels,dtype='int32')]=-row_sum
                
                
                if(C !=0.0):
                    binary2 = np.zeros(scores.shape)
                    binary2[scores>0.0] = 1.0                
                    binary2[scores<0.0] = -1.0
                else:
                    binary2 = 0
                    
                dscores1 = binary1
                dscores2 = binary2
                if(class_weighting=='average'):
                    gradW = np.dot((dscores1 + C*dscores2).transpose(),data)
                    gradW=gradW.transpose()
                    gradW = (1.0/N)*gradW
#                    gradW += gradW1 - gradW2
                else:
                    sample_weights_b=sample_weights_batch[batch_num]
                    gradW=np.dot((dscores1 + C*dscores2).transpose(),data*sample_weights_b[:,None])
                    gradW=gradW.transpose()
                    gradW=(1.0/numClasses)*gradW
#                    gradW += gradW1 - gradW2
                        
                if(np.sum(gradW**2)>G_clip_threshold):#gradient clipping
                    gradW = G_clip_threshold*gradW/np.sum(gradW**2)
                    
                if(update_type=='sgd'):
                    W = W - eta*gradW         
                else:
                    W = W - eta*gradW
                    
                if(reg_type=='l2'):
                    W += -D*W*(eta)  
                    
                if(reg_type=='l1' or reg_type == 'M'):
                    u = u + D*eta
                    z[:] = W[:]
                    idx_plus = W>0
                    idx_minus = W<0
                    
                    W_temp = np.zeros(W.shape)
                    W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u+q[idx_plus]))
                    W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u-q[idx_minus]))
                    W[idx_plus]=W_temp[idx_plus]
                    W[idx_minus]=W_temp[idx_minus]
                    q=q+(W-z)
                
                if(np.sum(W**2)>W_clip_threshold):#gradient clipping
                    W = W_clip_threshold*W/np.sum(W**2)
            
            if(iter1%iter_eval==0):                    
                #once the W are calculated for each epoch we calculate the scores
                scores=xTrain.dot(W)
#                scores=scores-np.max(scores)
                N=scores.shape[0]
                correct_scores = scores[range(N),np.array(yTrain,dtype='int32')]
                mat = (scores.transpose()-correct_scores.transpose()).transpose() 
                mat = mat+1.0
                mat[range(N),np.array(yTrain,dtype='int32')] = 0.0
                thresh1 = np.zeros(mat.shape)
                thresh1[mat>0.0] = mat[mat>0.0] #for the SVM loss 
                
                f=0.0
                if(reg_type=='l2'):
                    f += D*0.5*np.sum(W**2) 
                if(reg_type=='l1'):
                    f += D*np.sum(np.abs(W))

                if(class_weighting=='average'):
                    f1 = C*np.sum(np.abs(scores)) + np.sum(thresh1)
                    f += (1.0/N)*f1 
                else:
                    f1 = C*np.sum(np.abs(scores)*sample_weights[:,None]) + np.sum(thresh1*sample_weights[:,None])
                    f+= (1.0/numClasses)*f1 
                    
                for i in range(numClasses):
                    #first term in objective function for margin
                    if(K_plus is not None):
                        w = W[0:(M-1),i]
                        f2 = np.dot(np.dot(K_plus,w),w)
                        f += ((0.5*E)/(numClasses))*f2  
                        #the second term in the objective function for margin
                    if(K_minus is not None):
                        f3 = np.dot(temp_mat[:,i],w)
                        f += -((0.5*E)/(numClasses))*f3
                if(verbose == True):        
                    print('iter1=%d, f=%0.3f'%(iter1,f))
                fvals[iter1]=f
                rel_error=np.abs(f_prev-f)/np.abs(f_prev)
                max_W = np.max(np.abs(W))
                W[np.abs(W)<1e-03*max_W]=0.0
                if(f<f_best):
                    f_best=f
                    W_best[:]=W[:]
                    max_W = np.max(np.abs(W))
                    W_best[np.abs(W_best)<1e-03*max_W]=0.0
                    iter_best=iter1
                else:
                    break
                f_prev=f      
 
            eta=eta_zero/np.power((iter1+1),1)
            
        fvals[iter1]=-1
        return W_best,f_best,iter_best,fvals
            
 
    def predict(self,data, xTrain, W_all, sample_indices, feature_indices, me_all, std_all, subset_all, conformal=False, qtestr=None,qt=None):
        #type=2 -> mode of all labels
        #type=1 -> average of all labels
        #type=3 -> concat of all labels
        types = self.combine_type
        kernel_type = self.kernel_type
        gamma = self.gamma
        n_components = self.n_components
        
        n_ensembles = feature_indices.shape[0]
        N = data.shape[0]  
        M = data.shape[1]
        
        numClasses = W_all[0].shape[1]
        label = np.zeros((N,)) 
            
        label_all_2=np.zeros((N,numClasses))
        for i in range(n_ensembles):                
#                print('testing PLM %d'%i)
            X1, X2 = self.select_(data, xTrain, kernel_type, subset_all[i], feature_indices[i,:], sample_indices[i,:])
            data1 = self.kernel_transform( X1 = X1, X2 = X2, kernel_type = kernel_type, n_components = n_components, gamma = gamma)
            
            # CONFORMAL rows - test columns - train
            if(conformal):
                data1=((data1*qt).T*qtestr).T
            #
            data1 = self.normalize_(data1,me_all[i],std_all[i])
            
            M = data1.shape[1]
            data1 = self.add_bias(data1)                                        
            
            W = W_all[i]  
            
            if(self.problem_type == 'classification'):
                scores = data1.dot(W)
                label_all_2 += label_all_2 + scores
        
        if(self.problem_type == 'classification'):
            label=np.argmax(label_all_2,axis=1)
            return label   
    
    def accuracy_classifier(self,actual_label,found_labels):
        acc=np.divide(np.sum(actual_label==found_labels)*100.0 , actual_label.shape[0],dtype='float64')
        if(acc<50):
            acc=100-acc
        return acc
    
    def accuracy_regressor(self,actual_label,found_labels):
        acc=np.divide(np.linalg.norm(actual_label - found_labels)**2 , actual_label.shape[0],dtype='float64')
        if(acc<50):
            acc=100-acc
        return acc
        
    
    def train_LSMCM(self, xTrain, yTrain, level, K_plus = None, K_minus = None, W = None):
        #min D(E|w|_1 + (1-E)*0.5*|W|_2^2) + C*\sum_i\sum_(j)|f_j(i)**2| + \sum_i\sum_(j_\neq y_i)(1-f_y_i(i) + f_j(i))**2
        #setting C = 0 gives us SVM
        # or when using margin term i.e., reg_type = 'M'
        #min D(E|w|_1) + (E)*0.5*\sum_j=1 to numClasses (w_j^T(K+ - K-)w_j) + C*\sum_i\sum_(j)|f_j(i)**2| + \sum_i\sum_(j_\neq y_i)(1-f_y_i(i) + f_j(i))**2
        #setting C = 0 gives us SVM with margin term
#        print('LSMCM Training')
#        print('reg_type=%s, algo_type=%s, problem_type=%s,kernel_type=%s'%(self.reg_type,self.algo_type,self.problem_type,self.kernel_type))
#        print('C1=%0.4f, C2=%0.4f, C3=%0.4f'%(self.C1,self.C2,self.C3))
        if(self.upsample1==True):
            xTrain,yTrain=self.upsample(xTrain,yTrain,new_imbalance_ratio=0.5,upsample_type=1)
            
        xTrain=self.add_bias(xTrain)
        
        M=xTrain.shape[1]
        N=xTrain.shape[0]
        numClasses=np.unique(yTrain).size
        verbose = False
        if(level==0):
            C = self.C1 #for loss function of MCM
            D = self.C2 #for L1 or L2 penalty
            E = self.C3 #for elastic net penalty or margin term
        else:
            C = self.C4 #for loss function of MCM 
            D = self.C2 #for L1 or L2 penalty
            E = self.C3 #for elastic net penalty since in combining the classifiers we use a linear primal classifier
            
        iterMax1 = self.iterMax1
        eta_zero = self.eta
        class_weighting = self.class_weighting
        reg_type = self.reg_type
        update_type = self.update_type
        tol = self.tol
        np.random.seed(1)
        
        if(W is None):
            W=0.001*np.random.randn(M,numClasses)
            W=W/np.max(np.abs(W))
        else:
            W_orig = np.zeros(W.shape)
            W_orig[:] = W[:]
        
        class_weights=np.zeros((numClasses,))
        sample_weights=np.zeros((N,))
        #divide the data into K clusters
    
        for i in range(numClasses):
            idx=(yTrain==i)           
            class_weights[i]=1.0/np.sum(idx)
            sample_weights[idx]=class_weights[i]
                        
        G_clip_threshold = 100
        W_clip_threshold = 500
        eta=eta_zero
                       
        scores = xTrain.dot(W) #samples X numClasses
        N = scores.shape[0]
        correct_scores = scores[range(N),np.array(yTrain,dtype='int32')]
        mat = (scores.transpose()-correct_scores.transpose()).transpose() 
        mat = mat+1.0
        mat[range(N),np.array(yTrain,dtype='int32')] = 0.0
        
        scores1  = np.zeros(scores.shape)
        scores1[:] = scores[:]
        scores1[range(N),np.array(yTrain,dtype='int32')] = -np.inf
        max_scores = np.max(scores1,axis =1)
        mat1 = 1 - correct_scores + max_scores
#        thresh1 = np.zeros(mat.shape)
#        thresh1[mat>0.0] = mat[mat>0.0] #for the SVM loss 
        #(1- f_yi + max_j neq yi f_j)^2
        f=0.0
        if(reg_type=='l2'):
            f += D*0.5*np.sum(W**2) 
        if(reg_type=='l1'):
            f += D*np.sum(np.abs(W))
            
        if(class_weighting=='average'):
            f1 = C*0.5*np.sum(scores**2) + 0.5*np.sum((mat1)**2)
            f += (1.0/N)*f1 
        else:
            f1 = C*0.5*np.sum((scores**2)*sample_weights[:,None]) + 0.5*np.sum((mat1**2)*sample_weights[:,None])
            f+= (1.0/numClasses)*f1 
        
        if(K_minus is not None):
            temp_mat = np.dot(K_minus,W_orig[0:(M-1),])        
        
        for i in range(numClasses):
            #add the term (E/2*numclasses)*lambda^T*K_plus*lambda for margin
            if(K_plus is not None):
                w = W[0:(M-1),i]
                f2 = np.dot(np.dot(K_plus,w),w)
                f+= ((0.5*E)/(numClasses))*f2  
             #the second term in the objective function
            if(K_minus is not None):
                f3 = np.dot(temp_mat[:,i],w)
                f+= -((0.5*E)/(numClasses))*f3
        
        
        iter1=0
        print('iter1=%d, f=%0.3f'%(iter1,f))
                
        f_best=f
        fvals=np.zeros((iterMax1+1,))
        fvals[iter1]=f_best
        W_best=np.zeros(W.shape)
        iter_best=iter1
        f_prev=f_best
        rel_error=1.0
#        f_prev_10iter=f
        
        if(reg_type=='l1' or reg_type =='en' or reg_type == 'M'):
            # from paper: Stochastic Gradient Descent Training for L1-regularized Log-linear Models with Cumulative Penalty
            if(update_type == 'adam' or update_type == 'adagrad' or update_type == 'rmsprop'):
                u = np.zeros(W.shape)
            else:
                u = 0.0
            q=np.zeros(W.shape)
            z=np.zeros(W.shape)
            all_zeros=np.zeros(W.shape)
        
        eta1=eta_zero 
        v=np.zeros(W.shape)
        v_prev=np.zeros(W.shape)    
        vt=np.zeros(W.shape)
        m=np.zeros(W.shape)
        vt=np.zeros(W.shape)
        
        cache=np.zeros(W.shape)
        eps=1e-08
        decay_rate=0.99
        mu1=0.9
        mu=mu1
        beta1 = 0.9
        beta2 = 0.999  
        iter_eval=10 #evaluate after every 10 iterations
        
        idx_batches, sample_weights_batch, num_batches = self.divide_into_batches_stratified(yTrain)
        while(iter1<iterMax1 and rel_error>tol):
            iter1=iter1+1            
            for batch_num in range(0,num_batches):
    #                batch_size=batch_sizes[j]
                test_idx=idx_batches[batch_num]
                data=xTrain[test_idx,]
                labels=yTrain[test_idx,] 
                N=labels.shape[0]
                scores=data.dot(W)
                correct_scores=scores[range(N),np.array(labels,dtype='int32')]#label_batches[j] for this line should be in the range [0,numClasses-1]
                mat=(scores.transpose()-correct_scores.transpose()).transpose() 
                mat=mat+1.0
                mat[range(N),np.array(labels,dtype='int32')]=0.0                
                
                scores1  = np.zeros(scores.shape)
                scores1[:] = scores[:]
                scores1[range(N),np.array(labels,dtype='int32')] = -np.inf
                max_scores = np.max(scores1,axis =1)
                max_scores_idx = np.argmax(scores1, axis = 1)
                mat1 = 1 - correct_scores + max_scores                
                
                dscores1 = np.zeros(mat.shape)
                dscores1[range(N),np.array(max_scores_idx,dtype='int32')] = mat1
                row_sum = np.sum(dscores1,axis=1)
                dscores1[range(N),np.array(labels,dtype='int32')] = -row_sum
                
                if(C !=0.0):
                    dscores2 = np.zeros(scores.shape)
                    dscores2[:] = scores[:]
                else:
                    dscores2 = 0
                    
                dscores1 = 2*dscores1
                dscores2 = 2*dscores2
                if(class_weighting=='average'):
                    gradW = np.dot((dscores1 + C*dscores2).transpose(),data)
                    gradW = gradW.transpose()
                    gradW = (0.5/N)*gradW
#                    gradW += gradW1 - gradW2
                else:
                    sample_weights_b = sample_weights_batch[batch_num]
                    gradW = np.dot((dscores1 + C*dscores2).transpose(),data*sample_weights_b[:,None])
                    gradW = gradW.transpose()
                    gradW = (0.5/numClasses)*gradW
#                    gradW += gradW1 - gradW2
                        
                if(np.sum(gradW**2)>G_clip_threshold):#gradient clipping
#                    print('clipping gradients')
                    gradW = G_clip_threshold*gradW/np.sum(gradW**2)
                    
                if(update_type=='sgd'):
                    W = W - eta*gradW
                else:
                    W = W - eta*gradW
                                            
                if(reg_type=='l2'):
                    if(update_type == 'adam'):
                        W += -D*W*(eta1/(np.sqrt(vt) + eps)) 
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        W += -D*W*(eta1/(np.sqrt(cache) + eps))
                    else:
                        W += -D*W*(eta)  
                
                if(reg_type=='l1' or reg_type == 'M'):
                    if(update_type=='adam'):
                        u = u + D*(eta1/(np.sqrt(vt) + eps))
                    elif(update_type == 'adagrad' or update_type =='rmsprop'):
                        u = u + D*(eta1/(np.sqrt(cache) + eps))
                    else:
                        u = u + D*eta
                    z[:] = W[:]
                    idx_plus = W>0
                    idx_minus = W<0
                    
                    W_temp = np.zeros(W.shape)
                    if(update_type=='adam' or update_type == 'adagrad' or update_type =='rmsprop'):
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u[idx_plus]+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u[idx_minus]-q[idx_minus]))                    
                    else:
                        W_temp[idx_plus]=np.maximum(all_zeros[idx_plus],W[idx_plus]-(u+q[idx_plus]))
                        W_temp[idx_minus]=np.minimum(all_zeros[idx_minus],W[idx_minus]+(u-q[idx_minus]))
                    
                    W[idx_plus]=W_temp[idx_plus]
                    W[idx_minus]=W_temp[idx_minus]
                    q=q+(W-z)
                    
                if(np.sum(W**2)>W_clip_threshold):#gradient clipping
#                    print('clipping normW')
                    W = W_clip_threshold*W/np.sum(W**2)
            
            if(iter1%iter_eval==0):                    
                #once the W are calculated for each epoch we calculate the scores
                scores=xTrain.dot(W)
#                scores=scores-np.max(scores)
                N=scores.shape[0]
                correct_scores = scores[range(N),np.array(yTrain,dtype='int32')]
                mat = (scores.transpose()-correct_scores.transpose()).transpose() 
                mat = mat+1.0
                mat[range(N),np.array(yTrain,dtype='int32')] = 0.0
#                thresh1 = np.zeros(mat.shape)
#                thresh1[mat>0.0] = mat[mat>0.0] #for the SVM loss 
                scores1  = np.zeros(scores.shape)
                scores1[:] = scores[:]
                scores1[range(N),np.array(yTrain,dtype='int32')] = -np.inf
                max_scores = np.max(scores1,axis =1)
                mat1 = 1 - correct_scores + max_scores
                
                f=0.0
                if(reg_type=='l2'):
                    f += D*0.5*np.sum(W**2) 
                if(reg_type=='l1'):
                    f += D*np.sum(np.abs(W))
                  
                if(class_weighting=='average'):
                    f1 = C*0.5*np.sum(scores**2) + 0.5*np.sum(mat1**2)
                    f += (1.0/N)*f1 
                else:
                    f1 = C*0.5*np.sum((scores**2)*sample_weights[:,None]) + 0.5*np.sum((mat1**2)*sample_weights[:,None])
                    f+= (1.0/numClasses)*f1 
                    
                for i in range(numClasses):
                    #first term in objective function for margin
                    if(K_plus is not None):
                        w = W[0:(M-1),i]
                        f2 = np.dot(np.dot(K_plus,w),w)
                        f += ((0.5*E)/(numClasses))*f2  
                        #the second term in the objective function for margin
                    if(K_minus is not None):
                        f3 = np.dot(temp_mat[:,i],w)
                        f += -((0.5*E)/(numClasses))*f3
                        
                if(verbose == True):        
                    print('iter1=%d, f=%0.3f'%(iter1,f))
                    
                fvals[iter1]=f
                rel_error=np.abs(f_prev-f)/np.abs(f_prev)
                max_W = np.max(np.abs(W))
                W[np.abs(W)<1e-03*max_W]=0.0
                
                if(f<f_best):
                    f_best=f
                    W_best[:]=W[:]
                    max_W = np.max(np.abs(W))
                    W_best[np.abs(W_best)<1e-03*max_W]=0.0
                    iter_best=iter1
                else:
                    break
                f_prev=f      
 
            eta=eta_zero/np.power((iter1+1),1)
            
        fvals[iter1]=-1
        return W_best,f_best,iter_best,fvals
