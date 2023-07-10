from load import *
from features_analysis.feature_analysis import *
from features_analysis.PCA import *
from features_analysis.LDA import *
from Gaussian_model.new_MVG_model import *
from Gaussian_model.MVG_density import *
from Gaussian_model.class_MVG import *
from evaluation_functions.evaluation import *
from validation.k_fold import *
from logistic_regression.logreg import logRegClass,quadLogRegClass
from svm.svm import SVMClass
import numpy as np
import matplotlib.pyplot as plt

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
    # plotSingle(DC, LTR,10)
    # plotCross(DC, LTR,10)
       
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
    #PCA_plot(DTR)
    
############################                          K-FOLD                  #########################################
    prior,Cfp,Cfn = (0.5,10,1)
# options={"K":5,
#           "pca":6,
#           "pi":0.5,
#           "costs":(1,10)}
                                ###### MVG  #####
# K = 5
# MVG_obj = GaussClass("MVG")
# NB_obj = GaussClass("NB")
# TCG_obj = GaussClass("TCG")
# TCGNB_obj = GaussClass("TCGNB") 

# mvg_pca6 = 0
# mvg_pca7 = 0
# mvg_pca8 = 0
# mvg_pca9 = 0
# mvg_pcaNone = 0

# for model in [MVG_obj,NB_obj,TCG_obj,TCGNB_obj]:
#     for pca in [6,7,8,9,None]:
#         options={"K":5,
#                   "pca":pca,
#                   "pi":0.5,
#                   "costs":(1,10)}
        
#         min_DCF, scores, labels = kfold(DTR, LTR, model,options)
#         if pca == 6:
#             mvg_pca6=min_DCF
#         if pca == 7:
#             mvg_pca7=min_DCF
#         if pca == 8:
#             mvg_pca8=min_DCF
#         if pca == 9:
#             mvg_pca9=min_DCF
#         if pca == None:
#             mvg_pcaNone=min_DCF
            
#         print(f"{model.name()} min_DCF con K = {K} , pca = {pca}: {min_DCF} ")
        
#     mvg_pca = np.array([mvg_pca6,
#     mvg_pca7 ,
#     mvg_pca8 ,
#     mvg_pca9 ,
#     mvg_pcaNone])
    
#     plt.xlabel("PCA dimensions")
#     plt.ylabel("DCF_min")
#     #plt.legend()
#     plt.title(model.name())
#     path= "plots/gaussian/"+str(model.name())
#     plt.plot(np.linspace(6,10,5),mvg_pca)
#     plt.savefig(path)
#     plt.show()
    
        

                            ####### LOG REG   #######
# K=  5
# piT = 0.1
# lamb  =np.logspace(-7, 2, num=9)
# for piT in [0.1,0.5,0.9]:
#     lr_pca6 = []
#     lr_pca7 = []
#     lr_pca8 = []
#     lr_pca9 = []
#     lr_pcaNone = []
#     for l in np.logspace(-6, 2, num=9):
#         #we saw that piT=0.1 is the best value
#             for pca in [6,7,8,9,None]:
#                 options={"K":5,
#                           "pca":pca,
#                           "pi":0.5,
#                           "costs":(1,10)}
#                 logObj = logRegClass(l,piT)
#                 min_DCF, scores, labels = kfold(DTR, LTR,logObj,options)
#                 print(f"Log Reg min_DCF con K = {K} , pca = {pca}, l = {l} , piT = {piT}: {min_DCF} ")
                
#                 if pca == 6:
#                     lr_pca6.append(min_DCF)
#                 if pca == 7:
#                     lr_pca7.append(min_DCF)
#                 if pca == 8:
#                     lr_pca8.append(min_DCF)
#                 if pca == 9:
#                     lr_pca9.append(min_DCF)
#                 if pca == None:
#                     lr_pcaNone.append(min_DCF)
    
#     plt.semilogx(lamb,lr_pca6, label = "PCA 6")
#     plt.semilogx(lamb,lr_pca7, label = "PCA 7")
#     plt.semilogx(lamb,lr_pca8, label = "PCA 8")
#     plt.semilogx(lamb,lr_pca9, label = "PCA 9")
#     plt.semilogx(lamb,lr_pcaNone, label = "No PCA")
    
#     plt.xlabel("Lambda")
#     plt.ylabel("DCF_min")
#     plt.legend()
#     if piT == 0.1:
#         path = "plots/logReg/DCF_su_lambda_piT_min"
#     if piT == 0.5:
#         path = "plots/logReg/DCF_su_lambda_piT_medium"
#     if piT == 0.9:
#         path = "plots/logReg/DCF_su_lambda_piT_max"
#     plt.title(piT)
#     plt.savefig(path)
#     plt.show()

                                        ####### QUAD LOG REG   #######
# K=  5
# piT = 0.1
# lamb  =np.logspace(-7, 2, num=9)
# for piT in [0.1,0.33,0.5,0.9]:
#     lr_pca6 = []
#     lr_pca7 = []
#     lr_pca8 = []
#     lr_pca9 = []
#     lr_pcaNone = []
#     for l in np.logspace(-6, 2, num=9):
#         #we saw that piT=0.1 is the best value
#             for pca in [6,7,8,9,None]:
#                 options={"K":5,
#                           "pca":pca,
#                           "pi":0.5,
#                           "costs":(1,10),
#                           "znorm":False}
#                 quadLogObj = quadLogRegClass(l, piT)
#                 min_DCF, scores, labels = kfold(DTR, LTR,quadLogObj,options)
#                 print(f"Log Reg min_DCF con K = {K} , pca = {pca}, l = {l} , piT = {piT}: {min_DCF} ")
                
#                 if pca == 6:
#                     lr_pca6.append(min_DCF)
#                 if pca == 7:
#                     lr_pca7.append(min_DCF)
#                 if pca == 8:
#                     lr_pca8.append(min_DCF)
#                 if pca == 9:
#                     lr_pca9.append(min_DCF)
#                 if pca == None:
#                     lr_pcaNone.append(min_DCF)
    
#     plt.semilogx(lamb,lr_pca6, label = "PCA 6")
#     plt.semilogx(lamb,lr_pca7, label = "PCA 7")
#     plt.semilogx(lamb,lr_pca8, label = "PCA 8")
#     plt.semilogx(lamb,lr_pca9, label = "PCA 9")
#     plt.semilogx(lamb,lr_pcaNone, label = "No PCA")
        
#     plt.xlabel("Lambda")
#     plt.ylabel("DCF_min")
#     plt.legend()
#     if piT == 0.1:
#         path = "plots/quadLogReg/znorm/DCF_su_lambda_piT_min_znorm"
#     if piT == 0.5:
#         path = "plots/quadLogReg/znorm/DCF_su_lambda_piT_medium_znorm"
#     if piT == 0.9:
#         path = "plots/quadLogReg/znorm/DCF_su_lambda_piT_max_znorm"
#     plt.title(piT)
#     plt.savefig(path)
#     plt.show()
        
                                            #######   SVM   #######
K=  5
piT = 0.1
for piT in [0.1,0.33,0.5,0.9]:
    svm_pca6 = []
    svm_pca7 = []
    svm_pca8 = []
    svm_pca9 = []
    svm_pcaNone = []
    C_values = np.logspace(-5, 2, num=8)
    for C in np.logspace(-5, 2, num=8):
        for K_svm in [1]:
        #we saw that piT=0.1 is the best value
            for pca in [6,8]:
                options={"K":5,
                          "pca":pca,
                          "pi":0.5,
                          "costs":(1,10),
                          "znorm":False}
                SVMObj = SVMClass(K_svm, C, piT)
                min_DCF, scores, labels = kfold(DTR, LTR,SVMObj,options)
                if min_DCF > 1: 
                    min_DCF = 1
                print(f"SVM min_DCF con K = {K} ,K_svm = {K_svm},  C = {C} , piT = {piT}, pca = {pca}: {min_DCF} ")
                
                if pca == 6:
                    svm_pca6.append(min_DCF)
                if pca == 7:
                    svm_pca7.append(min_DCF)
                if pca == 8:
                    svm_pca8.append(min_DCF)
                if pca == 9:
                    svm_pca9.append(min_DCF)
                if pca == None:
                    svm_pcaNone.append(min_DCF)
    
    plt.semilogx(C_values,svm_pca6, label = "PCA 6")
    plt.semilogx(C_values,svm_pca7, label = "PCA 7")
    plt.semilogx(C_values,svm_pca8, label = "PCA 8")
    plt.semilogx(C_values,svm_pca9, label = "PCA 9")
    plt.semilogx(C_values,svm_pcaNone, label = "No PCA")
        
    plt.xlabel("C")
    plt.ylabel("DCF_min")
    plt.legend()
    if piT == 0.1:
        path = "plots/svm/DCF_su_C_piT_min"
    if piT == 0.33:
        path = "plots/svm/DCF_su_C_piT_033"
    if piT == 0.5:
        path = "plots/svm/DCF_su_C_piT_medium"
    if piT == 0.9:
        path = "plots/svm/DCF_su_C_piT_max"
    
    plt.title(piT)
    plt.savefig(path)
    plt.show()
    


############################                     MODEL BUILDING         ############################################## 

                                    ############      MVG      ##################
# log_pred_MVG = MVG_approach(DTR, LTR, 0.5, DTE, LTE)
# acc_MVG,_= accuracy(log_pred_MVG,LTE)
# inacc_MVG = 1-acc_MVG
# print("Error rate MVG no PCA : "+str(inacc_MVG*100))

# log_pred_MVG = MVG_approach(DP, LTR, 0.5, DTEP, LTE)
# acc_MVG,_= accuracy(log_pred_MVG,LTE)
# inacc_MVG = 1-acc_MVG
# print("Error rate MVG PCA con mp = "+ str(mp)+ " ml: "+ str(ml) +" : "+str(inacc_MVG*100))

# # log_pred_MVG = MVG_approach(DW, LTR, 0.5, DTEW, LTE)
# # acc_MVG,_= accuracy(log_pred_MVG,LTE)
# # inacc_MVG = 1-acc_MVG
# # print("Error rate MVG LDA con ml = "+str(ml)+ " : "+ str(inacc_MVG*100))

# log_pred_MVG = MVG_approach(DPW, LTR, 0.5, DTEPW, LTE)
# acc_MVG,_= accuracy(log_pred_MVG,LTE)
# inacc_MVG = 1-acc_MVG
# print("Error rate MVG LDA + PCA con m = "+ str(mp)+ " : "+str(inacc_MVG*100))

# print("------------------------")

# ##NB
# log_pred_NB = NB_approach(DTR, LTR, 0.5, DTE, LTE)
# acc_NB,_= accuracy(log_pred_NB,LTE)
# inacc_NB = 1-acc_NB
# print("Error rate MVG_NB no PCA : "+str(inacc_NB*100))

# log_pred_NB = NB_approach(DP, LTR, 0.5, DTEP, LTE)
# acc_NB,_= accuracy(log_pred_NB,LTE)
# inacc_NB = 1-acc_NB
# print("Error rate MVG_NB PCA con m = "+ str(mp)+ " : "+str(inacc_NB*100))

# log_pred_NB = NB_approach(DPW, LTR, 0.5, DTEPW, LTE)
# acc_NB,_= accuracy(log_pred_NB,LTE)
# inacc_NB = 1-acc_NB
# print("Error rate MVG_NB LDA+PCA con m = "+ str(mp)+ " : "+str(inacc_NB*100))

# print("------------------------")

# #TCG
# log_pred_TCG = TCG_approach(DTR, LTR, 0.5, DTE, LTE)
# acc_TCG,_= accuracy(log_pred_TCG,LTE)
# inacc_TCG = 1-acc_TCG
# print("Error rate MVG_TCG no PCA : "+str(inacc_TCG*100))

# log_pred_TCG = TCG_approach(DP, LTR, 0.5, DTEP, LTE)
# acc_TCG,_= accuracy(log_pred_TCG,LTE)
# inacc_TCG = 1-acc_TCG
# print("Error rate MVG_TCG PCA con m = "+ str(mp)+ " : "+str(inacc_TCG*100))


# log_pred_TCG = TCG_approach(DPW, LTR, 0.5, DTEPW, LTE)
# acc_TCG,_= accuracy(log_pred_TCG,LTE)
# inacc_TCG = 1-acc_TCG
# print("Error rate MVG_TCG PCA + LDA con m = "+ str(mp)+ " : "+str(inacc_TCG*100))

# print("------------------------")

                    
                ####################                    LOG REG                     ########################
# logObj = logRegClass(DTR, LTR, 0.1)
# logObj.train();
# lp_pred,lg_llr = logObj.test(DTE)
# acc,_ = accuracy(lp_pred,LTE)
# DCF_min,_,_ = DCF_min_impl(lg_llr, LTE, prior, Cfp, Cfn)
# print(100-acc*100)
# print("LR DCF_min: "+str(DCF_min))


#####################        COST EVALUATION AND CALIBRATION         #######################
# prior,Cfp,Cfn = (0.5,10,1)
# means,S_matrices,_ = MVG_model(DTR,LTR) #3 means and 3 S_matrices -> 1 for each class (3 classes)
# ll0,ll1 = loglikelihoods(DTE,means,S_matrices)
# llr = ll1-ll0

# #COSTS and CALIBRATION with DCF
# post_prob = binary_posterior_prob(llr,prior,Cfn,Cfp)
# DCF_norm = DCF_norm_impl(llr, LTE, prior, Cfp, Cfn)
# print(DCF_norm)
# DCF_min,t_min,thresholds = DCF_min_impl(llr, LTE, prior, Cfp, Cfn)
# print("DCF_norm: "+str(DCF_norm))
# print("DCF_min: "+str(DCF_min)+" con t: "+str(t_min))

# #ROC and Bayes error
# thresholds = np.sort(post_prob)
# ROC_plot(thresholds, post_prob, LTE)
# Bayes_DCF,Bayes_DCF_min = Bayes_plot(llr,LTE)










    