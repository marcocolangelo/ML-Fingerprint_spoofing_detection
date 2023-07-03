import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
# from sklearn.discriminant_analysis  import LinearDiscriminantAnalysis

# def LDA_plot_skilearn(D,L):
#     # Creare un'istanza della classe LinearDiscriminantAnalysis
#     lda = LinearDiscriminantAnalysis()
    
#     # Adattare il modello ai dati
#     lda.fit(D, L)
    
#     # Proiettare i dati sulla direzione LDA
#     X_lda = lda.transform(D)
#     plt.hist(X_lda, bins=10)
#     plt.title('Istogramma delle caratteristiche del set di dati - Direzione LDA')
#     plt.show()

def vrow(col):
    return col.reshape((1,col.size))

def vcol(row):
    return row.reshape((row.size,1))

def createCenteredSWc(DC):      #for already centered data 
    C = 0
    for i in range(DC.shape[1]):
        C = C + np.dot(DC[:,i:i+1],(DC[:,i:i+1]).T)
    C = C/float(DC.shape[1])
    return C


def centerData(D):
    mu = D.mean(1)
    DC = D - vcol(mu)    #broadcasting applied
    return DC



def createSBSW(D,L):
    D0 = D[:,L==0]      #we take all the D rows but only the columns which correspond to the same columns in L where the value is 0   
    D1 = D[:,L==1]

    DC0 = centerData(D0)
    DC1 = centerData(D1)
 

    SW0 = createCenteredSWc(DC0)
    SW1 = createCenteredSWc(DC1)
   

    centeredSamples = [DC0,DC1] 
    allSWc = [SW0,SW1]
    
    samples = [D0,D1]
    mu = vcol(D.mean(1))

    SB=0
    SW=0

    for x in range(2):
        m = vcol(samples[x].mean(1))
        SW = SW + (allSWc[x]*centeredSamples[x].shape[1]) 
        SB = SB + samples[x].shape[1] * np.dot((m-mu),(m-mu).T)     #here we don't use centered samples because we apply a covariance between classed
                                                                    #and we take the mean off in the formula yet
        
    SB = SB/(float)(D.shape[1])
    SW = SW / (float)(D.shape[1])

    return SB,SW


def LDA1(D,L,m):

    SB, SW = createSBSW(D,L)        
    s,U = sp.linalg.eigh(SB,SW) #we use the scipy function which supports heigbert generalization eigenvectors scomposition 
    W = U[:,::-1][:,0:m]        #we must take the first m columns of U matrix

    return W

def LDA2(D,L,m):

    SB,SW = createSBSW(D,L)
    U,s,_ = np.linalg.svd(SW)
    P1 = np.dot(U,vcol(1.0/(s**0.5))*U.T)       #first transformation (whitening transformation) to apply a samples "CENTRIFICATION"
    SBtilde = np.dot(P1,np.dot(SB,P1.T))
    U,_,_ = np.linalg.svd(SBtilde)              
    P2 = U[:,0:m]                               #second tranformation (samples rotation) to obtain SB diagonalization

    return np.dot(P1.T,P2)



def LDA_impl(D,L,m) :
    W1 = LDA1(D,L,m)
    DW = np.dot(W1.T,D)     #D projection on sub-space W1
    return DW,W1

    # W2 = LDA2(D,L,m)
    # DW2 = np.dot(W2.T,D) 
    # plotCross(DW2,L)         #plot function adapted to 2-D representation (plot maybe is flipped because of rotation in the second tranformation?)
