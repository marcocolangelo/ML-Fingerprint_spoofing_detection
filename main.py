from load import *
from features_analysis.feature_analysis import *
from features_analysis.PCA import *
from features_analysis.LDA import *
from Gaussian_model.new_MVG_model import *
from Gaussian_model.MVG_density import *
from evaluation_functions.evaluation import *

if __name__=='__main__':
    (DTR,LTR), (DTE,LTE)=loadTrainTest('dataset\Train.txt','dataset\Test.txt')
    DTR = DTR.T
    LTR = LTR.T
    DTE = DTE.T
    
############################      DATA ANALISYS         ############################ 
    
    #plotSingle(DTR, LTR,10)
    # plotCross(DTR, LTR,10)
    DC = centerData(DTR)              #delete the mean component from the data
    
## plot the features
    #plotSingle(DC, LTR,10)
    #plotCross(DC, LTR,10)
       
    mp = 6
## PCA implementation
    DP,P = PCA_impl(DTR, mp)        #try with 2-dimension subplot
    DTEP = np.dot(P.T,DTE)
    
    # plotCross(DP,LTR,m)        #plotting the data on a 2-D cartesian graph
    # plotSingle(DP, LTR,m)
    
    ml = 5
## LDA implementation
    DW,W = LDA_impl(DTR,LTR,ml)
    DTEW = np.dot(W.T,DTE)
    #plotCross(DW,LTR,m)
    #plotSingle(DW, LTR, m)
    
## LDA + PCA implementation
    DPW,W = LDA_impl(DP,LTR,ml)
    DTEPW = np.dot(W.T,DTEP)
    
## Pearson correlation
    #Pearson_corr(DTR, LTR)
    #plot_pearson_correlation(DTR, LTR,1)
    
## PCA and variance plot
    PCA_plot(DTR)
    
############################       MODEL EVALUATION         ############################ 

# ##MVG
# log_pred_MVG = MVG_approach(DTR, LTR, 0.5, DTE, LTE)
# acc_MVG,_= evaluation(log_pred_MVG,LTE)
# inacc_MVG = 1-acc_MVG
# print("Error rate MVG no PCA : "+str(inacc_MVG*100))

# log_pred_MVG = MVG_approach(DP, LTR, 0.5, DTEP, LTE)
# acc_MVG,_= evaluation(log_pred_MVG,LTE)
# inacc_MVG = 1-acc_MVG
# print("Error rate MVG PCA con mp = "+ str(mp)+ " ml: "+ str(ml) +" : "+str(inacc_MVG*100))

# # log_pred_MVG = MVG_approach(DW, LTR, 0.5, DTEW, LTE)
# # acc_MVG,_= evaluation(log_pred_MVG,LTE)
# # inacc_MVG = 1-acc_MVG
# # print("Error rate MVG LDA con ml = "+str(ml)+ " : "+ str(inacc_MVG*100))

# log_pred_MVG = MVG_approach(DPW, LTR, 0.5, DTEPW, LTE)
# acc_MVG,_= evaluation(log_pred_MVG,LTE)
# inacc_MVG = 1-acc_MVG
# print("Error rate MVG LDA + PCA con m = "+ str(mp)+ " : "+str(inacc_MVG*100))

# print("------------------------")

# ##NB
# log_pred_NB = NB_approach(DTR, LTR, 0.5, DTE, LTE)
# acc_NB,_= evaluation(log_pred_NB,LTE)
# inacc_NB = 1-acc_NB
# print("Error rate MVG_NB no PCA : "+str(inacc_NB*100))

# log_pred_NB = NB_approach(DP, LTR, 0.5, DTEP, LTE)
# acc_NB,_= evaluation(log_pred_NB,LTE)
# inacc_NB = 1-acc_NB
# print("Error rate MVG_NB PCA con m = "+ str(mp)+ " : "+str(inacc_NB*100))

# log_pred_NB = NB_approach(DPW, LTR, 0.5, DTEPW, LTE)
# acc_NB,_= evaluation(log_pred_NB,LTE)
# inacc_NB = 1-acc_NB
# print("Error rate MVG_NB LDA+PCA con m = "+ str(mp)+ " : "+str(inacc_NB*100))

# print("------------------------")

# #TCG
# log_pred_TCG = TCG_approach(DTR, LTR, 0.5, DTE, LTE)
# acc_TCG,_= evaluation(log_pred_TCG,LTE)
# inacc_TCG = 1-acc_TCG
# print("Error rate MVG_TCG no PCA : "+str(inacc_TCG*100))

# log_pred_TCG = TCG_approach(DP, LTR, 0.5, DTEP, LTE)
# acc_TCG,_= evaluation(log_pred_TCG,LTE)
# inacc_TCG = 1-acc_TCG
# print("Error rate MVG_TCG PCA con m = "+ str(mp)+ " : "+str(inacc_TCG*100))


# log_pred_TCG = TCG_approach(DPW, LTR, 0.5, DTEPW, LTE)
# acc_TCG,_= evaluation(log_pred_TCG,LTE)
# inacc_TCG = 1-acc_TCG
# print("Error rate MVG_TCG PCA + LDA con m = "+ str(mp)+ " : "+str(inacc_TCG*100))

# print("------------------------")


## Cost evaluation
prior,Cfp,Cfn = (0.5,10,1)
means,S_matrices,_ = MVG_model(DTR,LTR) #3 means and 3 S_matrices -> 1 for each class (3 classes)
ll0,ll1 = loglikelihoods(DTE,means,S_matrices)
llr = ll1-ll0

#evaluation with DCF
post_prob = binary_posterior_prob(llr,prior,Cfn,Cfp)
#DCF_norm = DCF_norm_impl(llr, LTE, prior, Cfp, Cfn)
#DCF_min,t_min,thresholds = DCF_min_impl(llr, LTE, prior, Cfp, Cfn)
thresholds = np.sort(post_prob)
ROC_plot(thresholds, post_prob, LTE)
Bayes_DCF,Bayes_DCF_min = Bayes_plot(llr,label)
print("DCF_norm: "+str(DCF_norm))
print("DCF_min: "+str(DCF_min)+" con t: "+str(t_min))






    