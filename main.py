from load import *
from features_analysis.feature_analysis import *
from features_analysis.PCA import *
from features_analysis.LDA import *
from Gaussian_model.new_MVG_model import *
from Gaussian_model.MVG_density import *

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
       
    m = 2
## PCA implementation
    DP = PCA_impl(DTR, m)  #try with 2-dimension subplot
    # plotCross(DP,LTR,m)        #plotting the data on a 2-D cartesian graph
    # plotSingle(DP, LTR,m)
    
    m = 1
## LDA implementation
    DW = LDA_impl(DTR,LTR,m)
    #plotCross(DW,LTR,m)
    #plotSingle(DW, LTR, m)
    
## Pearson correlation
    #Pearson_corr(DTR, LTR)
    #plot_pearson_correlation(DTR, LTR,1)
    
## PCA and variance plot
    PCA_plot(DTR)
    
############################       MODEL EVALUATION         ############################ 

log_pred_MVG = MVG_approach(DTR, LTR, 0.5, DTE, LTE)
acc_MVG,_= evaluation(log_pred_MVG,LTE)
inacc_MVG = 1-acc_MVG
print(inacc_MVG*100)

log_pred_NB = NB_approach(DTR, LTR, 0.5, DTE, LTE)
acc_NB,_= evaluation(log_pred_NB,LTE)
inacc_NB = 1-acc_NB
print(inacc_NB*100)

log_pred_TCG = TCG_approach(DTR, LTR, 0.5, DTE, LTE)
acc_TCG,_= evaluation(log_pred_TCG,LTE)
inacc_TCG = 1-acc_TCG
print(inacc_TCG*100)



    