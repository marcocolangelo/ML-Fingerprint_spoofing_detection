import numpy as np

# Carica il tuo dataset

def mcol(v):
    return v.reshape((v.size, 1))


def mrow(v):
    return v.reshape((1, v.size))

def znorm_impl(DTR, DTE):
    print(DTR.shape)
    mu_DTR = mcol(DTR.mean(1))
    std_DTR = mcol(DTR.std(1))

    DTR_z = (DTR - mu_DTR) / std_DTR
    DTE_z = (DTE - mu_DTR) / std_DTR
    print(DTR_z.shape)
    return DTR_z, DTE_z

def normalize_zscore(D, mu=[], sigma=[]):
   # print("D shape: "+str(D.shape))
    if mu == [] or sigma == []:
        mu = np.mean(D, axis=1)
        sigma = np.std(D, axis=1)
    ZD = D
    ZD = ZD - mcol(mu)
    ZD = ZD / mcol(sigma)
    #print("ZD shape: "+str(ZD.shape))
    return ZD, mu, sigma

