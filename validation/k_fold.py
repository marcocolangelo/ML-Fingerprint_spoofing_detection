import numpy
from evaluation_functions.evaluation import *
from Gaussian_model.new_MVG_model import *
from features_analysis.PCA import *
from features_analysis.z_norm import znorm_impl,normalize_zscore

# def k_fold_cross_validation(X, y, K, model,prior,Cfn,Cfp):
#     """
#     Performs K-Fold cross-validation on a dataset.
    
#     Parameters:
#     X: array-like, shape (n_samples, n_features).
#         Feature matrix.
#     y: array-like, shape (n_samples,)
#         Vector of targets.
#     K: int
#         Number of folds.
#     model: object with fit and predict methods.
#         Machine learning model to be evaluated.
#     metric: callable
#         Function to compute a performance metric between y_true and y_pred.
    
#     Returns:
#     scores: list of float
#         List of performance scores for each iteration of the cross validation.
#     """
#     n_samples = X.shape[0]
#     fold_size = n_samples // K
#     scores = []
    
#     for i in range(K):
#         # Suddividi il set di dati in set di addestramento e test
#         start = i * fold_size
#         end = start + fold_size
#         X_test = X[start:end]
#         y_test = y[start:end]
#         X_train = np.concatenate((X[:start], X[end:]))
#         y_train = np.concatenate((y[:start], y[end:]))
        
#         # Addestra il modello sui dati di addestramento
#         model.fit(X_train, y_train)
        
#         # Valuta le prestazioni del modello sui dati di test
#         llr = model.predict(X_test)
        
#         #COSTS and CALIBRATION with DCF
#         DCF_norm = DCF_norm_impl(llr, y_test, prior, Cfp, Cfn)
#         DCF_min,t_min,thresholds = DCF_min_impl(llr, y_test, prior, Cfp, Cfn)
#         print("DCF_norm: "+str(DCF_norm))
#         print("DCF_min: "+str(DCF_min)+" con t: "+str(t_min) +" per i: "+str(i))
        
#         scores.append(DCF_min)
    
#     return scores


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

# def k_fold_Gauss(dataset,label,K,prior,Cfp,Cfn):
#     N = int(dataset.shape[1]/K)
#     classifiers = [(MVG_llr, "Multivariate Gaussian Classifier"), (NB_llr, "Naive Bayes"), (TCG_llr, "Tied Covariance")]
    
#     for j, (c, cstring) in enumerate(classifiers):
#         nWrongPrediction = 0
#         numpy.random.seed(j)
#         indexes = numpy.random.permutation(dataset.shape[1])
#         DCF_norm = 0
#         for i in range(K):
    
#             idxTest = indexes[i*N:(i+1)*N]
    
#             if i > 0:
#                 idxTrainLeft = indexes[0:i*N]
#             elif (i+1) < K:
#                 idxTrainRight = indexes[(i+1)*N:]
    
#             if i == 0:
#                 idxTrain = idxTrainRight
#             elif i == K-1:
#                 idxTrain = idxTrainLeft
#             else:
#                 idxTrain = numpy.hstack([idxTrainLeft, idxTrainRight])
            
#             DTR = dataset[:, idxTrain]
#             LTR = label[idxTrain]
#             DTE = dataset[:, idxTest]
#             LTE = label[idxTest]
#             #COSTS and CALIBRATION with DCF
#             llr = c(DTR,LTR,DTE)
#             DCF_norm += DCF_norm_impl(llr, LTE, prior, Cfp, Cfn)
#             #DCF_min,t_min,thresholds += DCF_min_impl(llr, y_test, prior, Cfp, Cfn)
    
#         DCF_mean = DCF_norm/K
        
#         print(f"{cstring} results:\DCF_norm: {DCF_mean, 1}%\n")