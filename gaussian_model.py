import numpy as np

def vcol(v):
    return v.reshape((v.size,1))
def vrow(v):
    return v.reshape((1,v.size))

def cov(dataset,mu):
    dmu=dataset-mu
    sigma=np.dot(dmu,(dmu.T))*(1/dataset.shape[1])
    return sigma

def loglikelihood2(xnd,mu,sigma):
    M=xnd.shape[0]
    value1=-(M/2)*np.log(2*np.pi)
    value2=-0.5*np.linalg.slogdet(sigma)[1]
    inv=np.linalg.inv(sigma)
    value3=-0.5*np.dot((xnd-mu).T,np.dot(inv,(xnd-mu))).ravel()
    return value1+value2+value3

def logpdf_GAU_ND(xnd,mu,sigma):
    Y=[]
    logN=0
    for i in range(xnd.shape[1]):
        Y.append(loglikelihood2(xnd[:,i:i+1],mu,sigma))
    return np.array(Y).ravel()

def loglikelihood(XND,mu,sigma):
    logi=logpdf_GAU_ND(XND,mu,sigma)
    logn=0
    for i in range(logi.shape[0]):
        logn+=logi[i]
    return logn

def gaussian_model(dataset,labels):
    mu=vcol(dataset.mean(1))
    sigma=cov(dataset,mu)
    ll=loglikelihood(dataset,mu,sigma)