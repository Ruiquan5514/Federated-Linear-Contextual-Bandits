import math
import random
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import pdb
import minVar
import Functions as F
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
#import movielens
import pandas as pd
Max = 100000
N = 100000


M = 100
K = 10
D = 3



Theta = np.zeros([K,D])
for a in range(K):
    Theta[a,0] = 1
c = np.zeros([M,K,D])
for i in range(M):
    a0 = np.random.randint(K-1)
    for a in range(K):
        tep = []
        if a == a0:
            for s in range(D):
                if s == 0:
                    tep.append(random.random()*0.1+0.7)
                else:
                    tep.append(random.random()*0.6/(D-1))
        else:
            for s in range(D):
                if s == 0:
                    tep.append(random.random()*0.1+0.5)
                else:
                    tep.append(random.random()*0.8/(D-1))
        c[i,a] = np.array(tep)


l = 10
L = 0
for i in range(M):
    for a in range(K):
        if c[i,a].dot(c[i,a]) < l: l = c[i,a].dot(c[i,a])
        if c[i,a].dot(c[i,a]) > L: L = c[i,a].dot(c[i,a])