import numpy
from evaluation_functions.evaluation import *
from Gaussian_model.new_MVG_model import *
from features_analysis.PCA import *
from features_analysis.z_norm import znorm_impl,normalize_zscore


def kfold(D, L,classifier, options):
        
        K = options["K"]
        pca = options["pca"]
        pi = options["pi"]
        (cfn, cfp) = options["costs"]
        znorm = options["znorm"]
        
        samplesNumber = D.shape[1]
        N = int(samplesNumber / K)
        
        numpy.random.seed(seed=0)
        indexes = numpy.random.permutation(D.shape[1])
        
        scores = numpy.array([])
        labels = numpy.array([])
       
        
        for i in range(K):
            idxTest = indexes[i*N:(i+1)*N]
            
            idxTrainLeft = indexes[0:i*N]
            idxTrainRight = indexes[(i+1)*N:]
            idxTrain = numpy.hstack([idxTrainLeft, idxTrainRight])
            
            DTR = D[:, idxTrain]
            LTR = L[idxTrain]   
            DTE = D[:, idxTest]
            LTE = L[idxTest]
            
            #zed-normalizes the data with the mu and sigma computed with DTR
           # DTR, mu, sigma = normalize_zscore(DTR)
           # DTE, mu, sigma = normalize_zscore(DTE, mu, sigma)
            
            
            if znorm == True:
                DTR,mu,sigma= normalize_zscore(DTR)
                DTE,_,_ = normalize_zscore(DTE,mu,sigma)
            
            if pca is not None: #PCA needed
                DTR, P = PCA_impl(DTR, pca)
                DTE = numpy.dot(P.T, DTE)
                
                
            classifier.train(DTR, LTR)
            
            scores_i = classifier.compute_scores(DTE)
            #print("SMV score: ")
            #print(scores_i)
            scores = numpy.append(scores, scores_i)
            labels = numpy.append(labels, LTE)
            
            
        labels = np.array(labels,dtype=int)
        min_DCF,_,_ = DCF_min_impl(scores, labels, pi, cfp, cfn)
     
        return min_DCF, scores, labels

