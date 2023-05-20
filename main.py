from load import *
from gaussian_model import *
from mvg_model import *

if __name__=='__main__':
    (DTR,LTR), (DTE,LTE)=load('project\Train.txt','project\Test.txt')
    #gaussian_model(DTR,LTR)
    LTR=np.reshape(LTR,(1,LTR.shape[0]))
    LTE=np.reshape(LTE,(1,LTE.shape[0]))
    MVG_log(DTR,LTR,DTE,LTE)
    #naive_bayes_classifier(DTR,LTR,DTE,LTE)