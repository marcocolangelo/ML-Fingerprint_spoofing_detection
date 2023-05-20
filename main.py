from load import *
from Gaussian_model.lab_05 import *

if __name__=='__main__':
    (DTR,LTR), (DTE,LTE)=load('dataset\Train.txt','dataset\Test.txt')
   
    #qua devi ancora ruotare le matrici di DTR e rendere LTR una colonna
    #qui sotto ho provato ma non sono sicuro che sia giusto fare così
    DTR = DTR.T
    LTR = LTR.T
    
    pred_MVG = MVG_approach(DTR,LTR,DTE,1/3)
    
   
    