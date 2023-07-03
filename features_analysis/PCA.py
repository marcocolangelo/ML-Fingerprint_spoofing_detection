import numpy as np
import matplotlib.pyplot as plt

def vrow(col):
    return col.reshape((1, col.size))

def vcol(row):
    return row.reshape((row.size, 1))


def createCov(D, mu):
    mu = D.mean(1)
    C = 0
    for i in range(D.shape[1]):
        C += np.dot(D[:, i:i+1] - mu, (D[:, i:i+1] - mu).T)
    
    C /= float(D.shape[1])
    return C

def createCenteredCov(DC):
    C = 0
    for i in range(DC.shape[1]):
        C += np.dot(DC[:, i:i+1], DC[:, i:i+1].T)
    
    C /= float(DC.shape[1])
    return C

def centerData(D):
    mu = D.mean(1)
    DC = D - vcol(mu)
    return DC

def createP(C, m):
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    
    return P

def PCA_impl(D, m):
    DC = centerData(D)
    C = createCenteredCov(DC)
    P = createP(C, m)
    DP = np.dot(P.T, D)
    
    return DP,P

def PCA_plot(D):
    # Calcola i valori propri della matrice di covarianza
    DC = centerData(D)
    C = createCenteredCov(DC)
    eigenvalues, _ = np.linalg.eigh(C)
    
    # Ordina i valori propri in ordine decrescente
    eigenvalues = eigenvalues[::-1]
    
    # Calcola la varianza spiegata per ogni componente principale
    explained_variance = eigenvalues / np.sum(eigenvalues)
    y_min, y_max = plt.ylim()
    y_values = np.linspace(y_min, y_max, 20)
    plt.yticks(y_values)
    plt.xlim(right=9)
    # Creare un grafico della varianza spiegata
    plt.plot(np.cumsum(explained_variance))
    plt.xlabel('Numero di componenti')
    plt.ylabel('Varianza spiegata cumulativa')
    plt.grid()
    plt.show()
