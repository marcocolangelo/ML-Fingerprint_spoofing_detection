import numpy as np
import scipy as sc
from features_analysis.PCA import *
from features_analysis.z_norm import znorm_impl,normalize_zscore
from evaluation_functions.evaluation import * 

def vcol(vector):
    return vector.reshape((vector.shape[0], 1))

def vrow(vector):
    return vector.reshape((1, vector.shape[0]))


def num_samples(dataset):
    return dataset.shape[1];

#is a function absolutely similar to the classic linear LR but with a logistic_reg_calibration function added

class LRCalibrClass:
    def __init__(self,l,piT):
        self.l = l
        self.DTR=[]
        self.LTR=[]
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
   
    def logistic_reg_calibration(self,DTR, LTR, DTE):
        self.DTR  = DTR
        self.LTR = LTR
        self.nT = len(np.where(LTR == 1)[0])
        self.nF = len(np.where(LTR == 0)[0])
        _v, _J, _d = sc.optimize.fmin_l_bfgs_b(self.logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad=True)
        _w = _v[0:DTR.shape[0]]
        _b = _v[-1]
        calibration =  np.log(self.piT / (1 - self.piT))
        self.b = _b
        self.w = _w
        STE = np.dot(_w.T, DTE) + _b - calibration
        return _w, _b  

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
        
    
    
def kfold_calib(D, L,classifier, options,calibrated=False):
        
        K = options["K"]
        K=2
        pi = options["pi"]
        (cfn, cfp) = options["costs"]
        pca=options["pca"]
        znorm = options["znorm"]
        if calibrated==True:
            logObj=options["logCalibration"]
           
        samplesNumber = D.shape[1]
        N = int(samplesNumber / K)
        
        np.random.seed(seed=0)
        indexes = np.random.permutation(D.shape[1])
        actDCFtmp=0
        scores = ([])
        labels = ([])
        scores_CAL=([])
       
        
            
        for i in range(K):
            print("K fold: "+str(i))
            idxTest = indexes[i*N:(i+1)*N]
            
            idxTrainLeft = indexes[0:i*N]
            idxTrainRight = indexes[(i+1)*N:]
            idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
            
            DTR = D[:, idxTrain]
            LTR= L[idxTrain]   
            DTE = D[:, idxTest]
            LTE = L[idxTest]
            
            
            if pca is not None: #PCA needed
                DTR, P = PCA_impl(DTR, pca)
                DTE = np.dot(P.T, DTE)
                
          
            classifier.train(DTR, LTR)
            scores_i = classifier.compute_scores(DTE)
            
            scores = np.append(scores, scores_i)
            labels=np.append(labels,LTE)
           
        #plot the Bayes error BEFORE calibration   
        labels=np.array(labels,dtype=int)
        DCF_effPrior,DCF_effPrior_min = Bayes_plot(scores, labels)
        DCF_effPrior_return = DCF_effPrior
        DCF_effPrior_min_return = DCF_effPrior_min 
        print(DCF_effPrior)
        print(DCF_effPrior_min)
        #plot the ROC BEFORE calibration
        # post_prob = binary_posterior_prob(scores,pi,cfn,cfp)
        # thresholds = np.sort(post_prob)
        # ROC_plot(thresholds,post_prob,labels)
        
        DTRc = scores[:int(len(scores) * 0.7)]
        DTEc = scores[int(len(scores) * 0.7):]
        LTRc = labels[:int(len(labels) * 0.7)]
        LTEc = labels[int(len(labels) * 0.7):]
        estimated_w, estimated_b = logObj.logistic_reg_calibration(np.array([DTRc]), LTRc,
                                                          np.array([DTEc]))
        print("estimated_w: "+str(estimated_w))
        print("estimated_b: "+str(estimated_b))
        #(DTRc,LTRc), (DTEc,LTEc)= split_db_2to1(scores, labels)
        # estimated_b, estimated_w = logObj.train(DTRc,LTRc)
        
        scores_append = scores.reshape((1, scores.size))
        final_score = np.dot(estimated_w.T, scores_append) + estimated_b
        print("final score: ")
        print(final_score)
        DCF_effPrior,DCF_effPrior_min = Bayes_plot(final_score, labels)
        print(DCF_effPrior)
        print(DCF_effPrior_min)
        return DCF_effPrior_return,DCF_effPrior_min_return,scores,final_score,labels
        