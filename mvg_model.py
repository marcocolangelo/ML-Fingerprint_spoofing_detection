import numpy 
import scipy.special
import matplotlib.pyplot as plt

from gaussian_model import *


def mean_cov(dcls):
    mu=vcol(dcls.mean(1))
    cov=numpy.dot(dcls-mu,(dcls-mu).T)/dcls.shape[1]
    return mu,cov

def logpdf_GAU_ND(x,m,c) :
    M=x.shape[0]
    value_c=numpy.linalg.inv(c) #c represents the covariance matrix sigma
    log_c=numpy.linalg.slogdet(c)[1] #logdet
    second_value=0.5*log_c
    third_value=0.5*(numpy.dot((x-m).T,(numpy.dot(value_c, (x-m))))).ravel()
    first_value=(M/2)*(numpy.log(2*numpy.pi))
    return (-first_value-second_value-third_value)



def logpdf_GAU_MND(x,mu,C) :
    Y=[]
    for i in range(x.shape[1]): #shape[1] indica le colonne
        Y.append(logpdf_GAU_ND(x[:,i:i+1],mu,C)) # la notazione di x mi prende le colonne dalla iesima alla iesima+1 esclusa
    return numpy.array(Y).ravel()

def MVG_log(DTR,LTR,DTE,LTE):
    V=[]
    hlab_2={}
    for i in [0,1]:
        print(DTR.shape)
        print(LTR==2) ##problema
        dcls_2=DTR[:,LTR==i]
        hlab_2[i]=mean_cov(dcls_2)
    for i in range(2):
        mu,c=hlab_2[i]
        f=(logpdf_GAU_MND(DTE,mu,c))
        V.append(vrow(f)) 
    log_den=numpy.vstack(V)
    logSjoin=numpy.log(1/3)+log_den
    logSMarginal=scipy.special.logsumexp(logSjoin,axis=0)
    logSPost=logSjoin-logSMarginal
    SPost=numpy.exp(logSPost)
    predictions= numpy.argmax(logSPost,axis=0)==LTE
    predicted=predictions.sum()
    notpredicted=predictions.size-predicted
    acc=predicted/predictions.size
    return predicted,DTE.shape[1]

#naive bayes classifier  

def mean_cov_naive(dcls):
    mu=vcol(dcls.mean(1))
    cov=(numpy.dot(dcls-mu,(dcls-mu).T)/dcls.shape[1])*numpy.identity(dcls.shape[0])
    return mu,cov


   
def naive_bayes_classifier(DTR,LTR,DTE,LTE):
    V=[]
    hlab_naive=[]
    mu_naive=[]
    hlab_naive={} 
    for i in [0,1,2]:
        dcls_naive=DTR[:,LTR==i]
        hlab_naive[i]=mean_cov_naive(dcls_naive)
    for i in range(3):
        mu,s_naive=hlab_naive[i]
        f=(logpdf_GAU_MND(DTE,mu,s_naive))
        V.append(vrow(f)) 
    log_den=numpy.vstack(V)
    logSjoin=numpy.log(1/3)+log_den
    logSMarginal=scipy.special.logsumexp(logSjoin,axis=0)
    logSPost=logSjoin-logSMarginal
    SPost=numpy.exp(logSPost)
    predictions= numpy.argmax(logSPost,axis=0)==LTE
    predicted=predictions.sum()
    notpredicted=predictions.size-predicted
    acc=predicted/predictions.size
    return predicted,DTE.shape[1]
    



#tied covariance gaussian classifier

def mean_cov_tied(dcls):
    mu_tied=vcol(dcls.mean(1))
    cov=(numpy.dot(dcls-mu_tied,(dcls-mu_tied).T))
    return mu_tied,cov

hlab_=0
hlab_tied={}
def tied_classifier(DTR,LTR,DTE,LTE):
    Stied=0
    for i in range(3):
        dcls_tied=DTR[:,LTR==i]
        hlab_tied[i]=(mean_cov_tied(dcls_tied))
        Stied+=hlab_tied[i][1]
    V=[]
    Stied/=DTR.shape[1]
    for i in range(3):
        mu,c=hlab_tied[i]
        f=(logpdf_GAU_MND(DTE,mu,Stied))
        V.append(vrow(f)) #perchè ogni riga rappresenta una classe quindi per classe 0 riga 0
    log_den=numpy.vstack(V)
    #print(log_den)
    logSjoin=numpy.log(1/3)+log_den
    logSMarginal=scipy.special.logsumexp(logSjoin,axis=0)
    logSPost=logSjoin-logSMarginal
    SPost=numpy.exp(logSPost)
    predictions= numpy.argmax(logSPost,axis=0)==LTE
    predicted=predictions.sum()
    notpredicted=predictions.size-predicted
    acc=predicted/predictions.size
    #print("Accuracy working with tied classifier: ", acc)
    #print("Error wotking with tied classifier: ",(1-acc)*100)
    return predicted,DTE.shape[1]
    
