import sys
import numpy as np
import string
import scipy.special
import itertools


def load(path):
    f=open(path,'r')
    labels=[]
    features=[]
    for line in f:
        splitted=line.split(',')
        features.append([float(i) for i in splitted[:-1]])
        labels.append(splitted[-2:-1])
    labels=np.array(labels)
    features=np.array(features)
    print(labels)
    return features.T, labels 
        

