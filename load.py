import sys
import numpy as np
import string
import scipy.special
import itertools


def loadTrainTest(pathT,pathE):
    fT=open(pathT,'r')
    fE=open(pathE,'r')
    DTR=[]
    DTE=[]
    LTE=[]
    LTR=[]
    for line in fT:
        splitted=line.split(',')
        DTR.append([float(i) for i in splitted[:-1]])
        LTR.append(int(splitted[-1]))
    DTR=np.array(DTR)
    LTR=np.array(LTR)
    for line in fE:
        splitted=line.split(',')
        DTE.append([float(i) for i in splitted[:-1]])
        LTE.append(int(splitted[-1]))
    DTE=np.array(DTE)
    LTE=np.array(LTE)
    fT.close()
    fE.close()
    return (DTR, LTR),(DTE,LTE)


def mcol(v):
    return v.reshape((v.size, 1))

#qui trovi un caricamento di un solo file
def loadFile(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Spoofed fingerprint': 0,
        'Authentic fingerprint': 1,
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:4]
                attrs = mcol(np.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return np.hstack(DList), np.array(labelsList, dtype=np.int32)
        

