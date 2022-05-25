import math
import random
import numpy as np
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import pandas as pd
Max = 100000
N = 100000




M = 100
K = 30
D = 3



complete_rating = np.array(pd.read_csv('complete_ratings'))
complete_rating = complete_rating[:,1:]/5


model = NMF(n_components=D, init='nndsvda', max_iter=500)
W = model.fit_transform(complete_rating)
H = model.components_


W = W[:M,:]

Theta = np.transpose(H)
c = np.zeros([M,K,D])
l = 10
L = 0
for i in range(M):
    if W[i].dot(W[i])<l:
        l = W[i].dot(W[i])
    if W[i].dot(W[i])>L:
        L = W[i].dot(W[i])
    for a in range(K):
        c[i,a] = W[i]

kmeans = KMeans(n_clusters=K,random_state=0).fit(Theta)
Theta = kmeans.cluster_centers_

