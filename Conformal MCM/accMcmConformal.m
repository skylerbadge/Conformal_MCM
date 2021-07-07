function [testAccConf, trainAccConf] = accMcmConformal(xTrain,yTrain,xTest,yTest,lambda,kerTypeMCM,gam0,gam,Cpara)
    
        % set empirical cores
        edindex = abs(lambda)>1e-6;
        ed = [xTrain(edindex,:) yTrain(edindex)] ; %empirical data

        data = [xTrain yTrain];
        
        [m n] = size(data);
        data = sortrows(data,n); %sort rows to separate the classes. 
        a = ed(:,1:n-1);
        m2 = sum(data(:,n)+1)/2; % number of points in class m2 = -1 
        m1 = m - m2; 
        de = size(ed,1);
         
        y = data(:,n); 
        p = data(:,1:n-1); 
        
        % compute K0 kernel matrix 
        kernel = kerTypeMCM;
        K0 = zeros(m,m);
        firstW0 = zeros(m,m);
        for i = 1:m 
            for j = 1:m 
                K0(i,j) = kernelfunction(kernel, p(i,:), p(j,:),gam0);         % basic kernel 
                if i==j 
                    firstW0(i,j) = K0(i,j);
                else 
                    firstW0(i,j) = 0; 
                end 
            end 
        end 
        
        % Compute sub matrices
        
        K11 = K0(1:m1,1:m1);
        K12 = K0(1:m1,m1+1:m);
        K21 = K0(m1+1:m,1:m1);
        K22 = K0(m1+1:m,m1+1:m); 
        
        [m n] = size(p);

        % Compute K1 matrix
        k1matrix = zeros(m,de);
        for i = 1:m 
            for j = 1:de 
                k1matrix(i,j) = kernelfunction(kernel, p(i,:),a(j,:),gam);   
            end 
        end 
        
        % Compute W0 and B0
        e = ones(m,1);
        K1 = [e k1matrix]; 
        B0 = [(1/m1)*K11  zeros(m1,m2);zeros(m2,m1) (1/m2)*K22] - [ (1/m)*K11 (1/m)*K12 ; (1/m)*K21 (1/m)*K22]; 
        W0 = [firstW0]  - [(1/m1)*K11 zeros(m1,m2); zeros(m2,m1) (1/m2)*K22];
        
        %Solve eigen value problem and generate eigenvalues
        C =1e-6; D =1e-6 ;  
        [ralpha lam]  =  eig(K1'*B0*K1+ C*speye(de+1), K1'*W0*K1+D*speye(de+1)) ;

        % choose max eigenvalue
        max =0 ; maxid =0; 
        for i  = 1: de 
            if(lam(i,i) > max ) 
                max = lam(i,i); 
                maxid = i; 
            end 
           
        end
        
        % Calculate Q
        qt = K1 * ralpha(:,maxid);
        
        
        % Compute Conformal kernel matrix
        
        Kt = zeros(m);
        for  i = 1:m      
            for j = 1:m       
                Kt(i,j) = qt(i) * qt(j) * K0(i,j); 
            end 
        end 
        
        [ lambdaConf,bConf,hConf ] = mcm_linear_efs_conformal( p, y, kerTypeMCM, gam0, Cpara, qt );
        [~,trainAccConf] = mcmPredictConformal(p,y,p,y,Kt,lambdaConf,bConf);
        m = size(xTest,1);
        
        qtestr = zeros(1,m);
        rtestK= zeros(m,size(p,1)); 
        for  i = 1:m 
            qtestr(i) = ralpha(1,maxid);  
            for j = 1:de 
                qtestr(i)  = qtestr(i) + ralpha(j+1,maxid) * kernelfunction(kernel, xTest(i,:), a(j,:), gam); 
            end 
        end 
        for i = 1:m    
            for j = 1: size(p,1) 
                rtestK(i,j) = qtestr(i) * qt(j) * kernelfunction(kernel, xTest(i,:), p(j,:), gam0); 
            end     
        end 
        [~,testAccConf] = mcmPredictConformal(p,y,xTest,yTest,rtestK,lambdaConf,bConf);
end