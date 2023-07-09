import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import scipy.optimize as sc

import numpy as np
import scipy as sc

def vrow(col):
    return col.reshape((1,col.size))

def vcol(row):
    return row.reshape((row.size,1))

class quadLogRegClass:
    def __init__(self,l,piT):
        self.l = l
        
        #due of UNBALANCED classes (the spoofed has significantly more samples) we have to put a piT to apply this weigth
        self.piT = piT
    def gradient_test(DTR, LTR, l, pt):
        z=np.empty((LTR.shape[0]))    
        z=2*LTR-1
        def gradient(v):        #grad = np.array( [derivative_w(v),derivative_b(v)], dtype = np.float64)
            # print("derivative w: ", derivative_w(v).size)        # print("derivative b: ", derivative_b(v).size)
            w, b = v[0:-1], v[-1]
            #print("w shape: ", w.size)        #print("w", w)
            #derivata rispetto a w        first_term = l*w
     
            second_term=0        
            third_term = 0
     
            nt = DTR[:, LTR == 1].shape[1]
            nf = DTR.shape[1]-nt        
            #empirical prior        pt=nt/DTR.shape[1]
            first_term = l*w
            for i in range(DTR.shape[1]):            #S=DTR[:,i]
                S=np.dot(w.T,DTR[:,i])+b            
                ziSi = np.dot(z[i], S)
                if LTR[i] == 1:
                    internal_term = np.dot(np.exp(-ziSi),(np.dot(-z[i],DTR[:,i])))/(1+np.exp(-ziSi))                #print(1+np.exp(-ziSi))
                    second_term += internal_term            
                else :
                    internal_term_2 = np.dot(np.exp(-ziSi),(np.dot(-z[i],DTR[:,i])))/(1+np.exp(-ziSi))                
                    third_term += internal_term_2
             #derivative_w= first_term + (pi/nt)*second_term + (1-pi)/(nf) * third_term
                    derivative_w= first_term + (pt/nt)*second_term + (1-pt)/(nf) * third_term
                    #derivata rispetto a b
            first_term = 0                   
            second_term=0
    
            for i in range(DTR.shape[1]):            #S=DTR[:,i]
                S=np.dot(w.T,DTR[:,i])+b
                ziSi = np.dot(z[i], S)
                if LTR[i] == 1:                
                    internal_term = (np.exp(-ziSi) * (-z[i]))/(1+np.exp(-ziSi))
                    first_term += internal_term            
                else :
                    internal_term_2 = (np.exp(-ziSi) * (-z[i]))/(1+np.exp(-ziSi))                
                    second_term += internal_term_2
    
            #derivative_b= (pi/nt)*first_term + (1-pi)/(nf) * second_term        
            derivative_b= (pt/nt)*first_term + (1-pt)/(nf) * second_term        
            grad = np.hstack((derivative_w,derivative_b))
            return grad
        return gradient
        
    def quad_logreg_obj(self,v):
        loss = 0
        loss_c0 = 0
        loss_c1 = 0
        #for each possible v value in the current iteration (which corresponds to specific coord
        #obtained by the just tracked movement plotted from the actual Hessian and Gradient values and the previous calculated coord)
        #we extrapolate the w and b parameters to insert in the J loss-function
        def gradient_test(DTR, LTR, l, pt):
            z=np.empty((LTR.shape[0]))    
            z=2*LTR-1
            def gradient(v):        #grad = np.array( [derivative_w(v),derivative_b(v)], dtype = np.float64)
                # print("derivative w: ", derivative_w(v).size)        # print("derivative b: ", derivative_b(v).size)
                w, b = v[0:-1], v[-1]
                #print("w shape: ", w.size)        #print("w", w)
                #derivata rispetto a w        first_term = l*w
         
                second_term=0        
                third_term = 0
         
                nt = DTR[:, LTR == 1].shape[1]
                nf = DTR.shape[1]-nt        
                #empirical prior        pt=nt/DTR.shape[1]
                first_term = 0.5*l*w
                for i in range(DTR.shape[1]):            #S=DTR[:,i]
                    S=np.dot(w.T,DTR[:,i])+b            
                    ziSi = np.dot(z[i], S)
                    if LTR[i] == 1:
                        internal_term = np.dot(np.exp(-ziSi),(np.dot(-z[i],DTR[:,i])))/(1+np.exp(-ziSi))                #print(1+np.exp(-ziSi))
                        second_term += internal_term            
                    else :
                        internal_term_2 = np.dot(np.exp(-ziSi),(np.dot(-z[i],DTR[:,i])))/(1+np.exp(-ziSi))                
                        third_term += internal_term_2
                 #derivative_w= first_term + (pi/nt)*second_term + (1-pi)/(nf) * third_term
                        derivative_w= first_term + (pt/nt)*second_term + (1-pt)/(nf) * third_term
                        #derivata rispetto a b
                first_term = 0                   
                second_term=0
        
                for i in range(DTR.shape[1]):            #S=DTR[:,i]
                    S=np.dot(w.T,DTR[:,i])+b
                    ziSi = np.dot(z[i], S)
                    if LTR[i] == 1:                
                        internal_term = (np.exp(-ziSi) * (-z[i]))/(1+np.exp(-ziSi))
                        first_term += internal_term            
                    else :
                        internal_term_2 = (np.exp(-ziSi) * (-z[i]))/(1+np.exp(-ziSi))                
                        second_term += internal_term_2
        
                #derivative_b= (pi/nt)*first_term + (1-pi)/(nf) * second_term        
                derivative_b= (pt/nt)*first_term + (1-pt)/(nf) * second_term        
                grad = np.hstack((derivative_w,derivative_b))
                return grad
            return gradient
        
        w,b = v[0:-1],v[-1]
        n = self.DTR.shape[1]
        
        regularization = (self.l / 2) * np.sum(w ** 2) 
        
        #it's a sample way to apply the math transformation z = 2c - 1 
        for i in range(n):
            x = vcol(self.DTR[:,i])
            mat_x = np.dot(x,x.T)
            vec_x = vcol(np.hstack(mat_x))
            fi_x = np.vstack((vec_x,x))
            
            if (self.LTR[i:i+1] == 1):
                zi = 1
                loss_c1 += np.logaddexp(0,-zi * (np.dot(w.T,fi_x) + b))
                
            else:
                zi=-1
                loss_c0 += np.logaddexp(0,-zi * (np.dot(w.T,fi_x) + b))
        
        J = regularization + (self.piT / self.nT) * loss_c1 + (1-self.piT)/self.nF * loss_c0
        grad_funct = gradient_test(self.DTR, self.LTR, self.l, self.piT) 
        grad = grad_funct(v)
        return J,grad
    
    def train(self,DTR,LTR):
        self.DTR  = DTR
        self.LTR = LTR
        x0 = np.zeros(DTR.shape[0] + 1)
        
        self.nT = len(np.where(LTR == 1)[0])
        self.nF = len(np.where(LTR == 0)[0])
        
        print("sono dentro")
        
        # logRegObj = logRegClass(self.DTR, self.LTR, self.l) #I created an object logReg with logreg_obj inside
        
        #optimize.fmin_l_bfgs_b looks for secod order info to search direction pt and then find an acceptable step size at for pt
        #I set approx_grad=True so the function will generate an approximated gradient for each iteration
        params,f_min,_ = sc.optimize.fmin_l_bfgs_b(self.quad_logreg_obj, x0,approx_grad=True)
        print("sono uscito")
        #the function found the coord for the minimal value of logreg_obj and they conrespond to w and b
        
        self.b = params[-1]
        
        self.w = np.array(params[0:-1])
        
        self.S = []
        
        
        return self.b,self.w
    
    def compute_scores(self,DTE):
        #I apply the model just trained to classify the test set samples
        S = self.S
        for i in range(DTE.shape[1]):
            x = vcol(DTE[:,i:i+1])
            mat_x = np.dot(x,x.T)
            vec_x= vcol(np.hstack(mat_x))
            fi_x = np.vstack((vec_x,x))
            self.S.append(np.dot(self.w.T,fi_x) + self.b)
            
        pred = [1 if x > 0 else 0 for x in S] #I transform in 1 all the pos values and in 0 all the negative ones
        
        # acc = accuracy(S,LTE)
        
        # print(100-acc)
        
        
        return S


class logRegClass:
    def __init__(self,l,piT):
        self.l = l
        
        #due of UNBALANCED classes (the spoofed has significantly more samples) we have to put a piT to apply this weigth
        self.piT = piT
        
        
    def logreg_obj(self,v):
        loss = 0
        loss_c0 = 0
        loss_c1 = 0
        #for each possible v value in the current iteration (which corresponds to specific coord
        #obtained by the just tracked movement plotted from the actual Hessian and Gradient values and the previous calculated coord)
        #we extrapolate the w and b parameters to insert in the J loss-function
        
        
        w,b = v[0:-1],v[-1]
        w = vcol(w)
        n = self.DTR.shape[1]
        
        regularization = (self.l / 2) * np.sum(w ** 2) 
        
        #it's a sample way to apply the math transformation z = 2c - 1 
        for i in range(n):
            
            if (self.LTR[i:i+1] == 1):
                zi = 1
                loss_c1 += np.logaddexp(0,-zi * (np.dot(w.T,self.DTR[:,i:i+1]) + b))
            else:
                zi=-1
                loss_c0 += np.logaddexp(0,-zi * (np.dot(w.T,self.DTR[:,i:i+1]) + b))
        
        J = regularization + (self.piT / self.nT) * loss_c1 + (1-self.piT)/self.nF * loss_c0
        
        return J
   

    
    def train(self,DTR,LTR):
        self.DTR  = DTR
        self.LTR = LTR
        x0 = np.zeros(DTR.shape[0] + 1)
        
        self.nT = len(np.where(LTR == 1)[0])
        self.nF = len(np.where(LTR == 0)[0])
        
        print("sono dentro")
        
        # logRegObj = logRegClass(self.DTR, self.LTR, self.l) #I created an object logReg with logreg_obj inside
        
        #optimize.fmin_l_bfgs_b looks for secod order info to search direction pt and then find an acceptable step size at for pt
        #I set approx_grad=True so the function will generate an approximated gradient for each iteration
        params,f_min,_ = sc.optimize.fmin_l_bfgs_b(self.logreg_obj, x0,approx_grad=True)
        print("sono uscito")
        #the function found the coord for the minimal value of logreg_obj and they conrespond to w and b
        
        self.b = params[-1]
        
        self.w = np.array(params[0:-1])
        
        self.S = []
        
        
        return self.b,self.w
    
    def compute_scores(self,DTE):
        #I apply the model just trained to classify the test set samples
        S = self.S
        for i in range(DTE.shape[1]):
            x = DTE[:,i:i+1]
            x = np.array(x)
            x = x.reshape((x.shape[0],1))
            self.S.append(np.dot(self.w.T,x) + self.b)
        
        S = [1 if x > 0 else 0 for x in S] #I transform in 1 all the pos values and in 0 all the negative ones
        
        # acc = accuracy(S,LTE)
        
        # print(100-acc)
        llr = np.dot(self.w.T, DTE) + self.b
        
        return llr
        
        
        
    
   


# if __name__ == "__main__":
#     #Numerical optimizations tries
#     x,f_min,d = sc.optimize.fmin_l_bfgs_b(f, (0,0),approx_grad=True);
    
#     x2,f_min2,d2 = sc.optimize.fmin_l_bfgs_b(f_2, (0,0))
    
#     #binary classificator with Logistic Regression applied to the Iris dataset
#     l = 0.000001
#     D, L = load_iris_binary()
#     (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

# def logreg_impl(DTR,LTR,DTE,l):
#         x0 = np.zeros(DTR.shape[0] + 1)
        
        
#         logRegObj = logRegClass(DTR, LTR, l) #I created an object logReg with logreg_obj inside
        
#         #optimize.fmin_l_bfgs_b looks for secod order info to search direction pt and then find an acceptable step size at for pt
#         #I set approx_grad=True so the function will generate an approximated gradient for each iteration
#         params,f_min,_ = sc.optimize.fmin_l_bfgs_b(logRegObj.logreg_obj, x0,approx_grad=True)
        
#         #the function found the coord for the minimal value of logreg_obj and they conrespond to w and b
        
#         b = params[-1]
        
#         w = np.array(params[0:-1])
        
#         S = []
        
#         #I apply the model just trained to classify the test set samples
#         for i in range(DTE.shape[1]):
#             x = DTE[:,i:i+1]
#             x = np.array(x)
#             x = x.reshape((x.shape[0],1))
#             S.append(np.dot(w.T,x) + b)
        
#         S = [1 if x > 0 else 0 for x in S] #I transform in 1 all the pos values and in 0 all the negative ones
        
#         # acc = accuracy(S,LTE)
        
#         # print(100-acc)
        
#         return S

if __name__ == "__main__":
    # Genera un set di dati di classificazione sintetico
    X, y = make_classification(n_features=4, random_state=0)
    
    # Suddividi il set di dati in set di addestramento e test
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    
    # Crea un'istanza della classe quadLogRegClass con l e piT impostati su valori di esempio
    clf = quadLogRegClass(l=0.01, piT=0.5)
    
    # Addestra il modello sui dati di addestramento
    clf.train(X_train.T, y_train)
    
    # Calcola i punteggi previsti dal modello per i dati di test
    scores = clf.compute_scores(X_test.T)
    
    # Calcola le etichette previste dal modello per i dati di test
    y_pred = [1 if s > 0 else 0 for s in scores]
    
    # Calcola l'accuratezza del modello sui dati di test
    accuracy = np.mean(y_pred == y_test)
    print(f'Accuracy: {accuracy:.2f}')
