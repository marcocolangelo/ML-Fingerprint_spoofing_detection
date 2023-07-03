import numpy as np
import matplotlib.pyplot as plt

def vrow(col):
    return col.reshape((1,col.size))

def vcol(row):
    return row.reshape((row.size,1))


def createCov(D):   
    mu = 0
    C = 0
    mu = D.mean(1)
    for i in range(D.shape[1]):
        C = C + np.dot(D[:, i:i+1] - mu, (D[:, i:i+1] - mu).T)  #scalar product using numpy
        #with this formule we have just centered the data (PCA on NON CENTERED DATA is quite an unsafe operation) 
    
    C = C / float(D.shape[1])   #where the divider is the dimension N of our data 
    return C

def createCenteredCov(DC):      #for centered data yet
    C = 0
    for i in range(DC.shape[1]):
        C = C + np.dot(DC[:,i:i+1],(DC[:,i:i+1]).T)
    C = C/float(DC.shape[1])
    return C

def centerData(D):
    mu = D.mean(1)
    DC = D - vcol(mu)    #broadcasting applied
    return DC

#logpdf_GAU_ND algorithm for an array(just one sample at time)
def logpdf_GAU_ND_1Sample(x,mu,C):  
    #it seems that also for just one row at time we should use mu and C of the whole matrix
    xc = x-mu
    M = x.shape[0]
    logN = 0
    const = - 0.5 * M *np.log(2*np.pi)
    log_determ = np.linalg.slogdet(C)[1]
    lamb = np.linalg.inv(C)
    third_elem = np.dot(xc.T,np.dot(lamb,xc)).ravel()
    logN = const - 0.5 * log_determ - 0.5*third_elem

    return logN

def logpdf_GAU_ND(X,mu,C):          #logpdf_GAU_ND algorithm for a 2-D matrix
    logN = []
    #print("Dim di X in logpdf_GAU_ND: "+str(X.shape))
    for i in range(X.shape[1]):
        #[:,i:i+1] notation allows us to take just the i-th column at time
        #remember that with this notation we mean [i,i+1) (left-extreme not included)
        logN.append(logpdf_GAU_ND_1Sample(X[:,i:i+1],mu,C)) 
    return np.array(logN).ravel()


def loglikelihood(X,m,C):
    logN = 0
    logi = logpdf_GAU_ND(X, m, C)
    for i in range(logi.shape[0]):
        #[:,i:i+1] notation allows us to take just the i-th column at time
        #remember that with this notation we mean [i,i+1) (left-extreme not included)
        
        logN += logi[i]
        
    return logN



def conf_plot(X,m,C):

    plt.figure()
    plt.hist(X.ravel(),bins=50,density=True)
    XPlot=np.linspace(-8,12,1000)
    plt.plot(XPlot.ravel(),np.exp(logpdf_GAU_ND(vrow(XPlot), m, C)))
    
# Prova funzioni :
    
#    X1D = np.load("./X1D.npy") 
#    XND = np.load("./XND.npy")
   
#    X1D_cent=centerData(X1D)
#    XND_cent = centerData(XND)
#    m_ML = XND.mean(1)
#    m_ML = vcol(m_ML)
#    m_ML2 = vcol(X1D.mean(1))
#    C_ML = createCenteredCov(XND_cent)
#    C_ML2= createCenteredCov(X1D_cent)
   
#    p = logpdf_GAU_ND(XND_cent, m_ML, C_ML)
   
#    ll = loglikelihood(XND,m_ML,C_ML)
#    ll2 = loglikelihood(X1D,m_ML2,C_ML2)
   
#    conf_plot(X1D,m_ML2,C_ML2)
   
