from load import *
from Gaussian_model.lab_05 import *

if __name__=='__main__':
    (DTR,LTR), (DTE,LTE)=load('dataset/Train.txt','dataset/Test.txt')
   
    pred_MVG = MVG_approach(DTR,LTR,DTE,1/3)
    
   
    