import sys
import numpy as np
import string
import scipy.special
import itertools


def load(pathT,pathE):
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
        

