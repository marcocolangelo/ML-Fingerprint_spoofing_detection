import numpy as np
import matplotlib.pyplot as plt

def loadFile(fileName):
    matrix= np.zeros((150,4))
    labels = []
    hLabels = {
        'Iris-setosa':0,
        'Iris-versicolor':1,
        'Iris-virginica':2
    }
    
    with open(fileName,'r') as f:
        for i,riga in enumerate(f):
            
            elementi=riga.split(',')[0:5]
            label=elementi.pop()
            label=hLabels[label[:-1]]
            labels.append(label)

            for j,elemento in enumerate(elementi):
                matrix[i][j] = float(elemento)

    return matrix.T,np.array(labels,dtype=np.int32)   #we want a 4x150 but initially we had a 150x4

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

#logpdf_GAU_ND algorithm for an array(just one sample at time)
def logpdf_GAU_ND_1Sample(x,mu,C):  
    #it seems that also for just one row at time we should use mu and C of the whole matrix
    xc = x-mu
    M = x.shape[0]
    logN = 0
    const = - (M/2*np.log(2*np.pi))
    log_determ = np.linalg.slogdet(C)[1]
    lamb = np.linalg.inv(C)
    third_elem = np.dot(xc.T,np.dot(lamb,xc)).ravel()
    logN = const -0.5 * log_determ - 0.5*third_elem

    return logN

def logpdf_GAU_ND(X,mu,C):          #logpdf_GAU_ND algorithm for a 2-D matrix
    logN = []
    for i in range(X.shape[1]):
        #[:,i:i+1] notation allows us to take just the i-th column at time
        #remember that with this notation we mean [i,i+1) (left-extreme not included)
        logN.append(logpdf_GAU_ND_1Sample(X[:,i:i+1],mu,C)) 
    return np.array(logN).ravel()
    

# def prova_algoritmo():
   
#     Xplot = np.linspace(-8,12, 1000)
#     m = np.load('./muND.npy')
#     C = np.load('./CND.npy')
    
#     XND = np.load("./XND.npy")
#     sol = np.load("./llND.npy")
#     prov=logpdf_GAU_ND(XND, m, C);
    
    
#     plt.figure()
#     plt.plot(Xplot.ravel(),np.exp(logpdf_GAU_ND(vrow(Xplot), m, C)))
#     plt.show()
#     return sol,prov


# if __name__ == "__main__":
#     [D,L] = loadFile('D:\Desktop\I ANNO LM\II SEMESTRE\Machine Learning and Pattern recognition\ML - Lab\Lab04\iris.csv')
#     mu = D.mean(1)
#     mu = vcol(mu)  #this make mu a (M,1) array
#     C = createCov(D)
    
    
    
    
    
#     #the function below tries the logpdf_GAU_ND alg with a fake dataset in input 
#     soluzione,prova = prova_algoritmo()
    
    
#     print(np.abs(soluzione-prova).max())
    
    
    