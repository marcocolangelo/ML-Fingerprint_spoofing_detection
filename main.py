from load import *
from features_analysis.feature_analysis import *
from features_analysis.PCA import *
from features_analysis.LDA import *

if __name__=='__main__':
    (DTR,LTR), (DTE,LTE)=loadTrainTest('dataset\Train.txt','dataset\Test.txt')
    DTR = DTR.T
    LTR = LTR.T
    
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
    