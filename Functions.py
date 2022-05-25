import math
import random
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import pdb
import minVar
Max = 100000
N = 100000


def PullArm(a, x, Theta, number = 1):
    reward = number*np.dot(Theta[a,:],x)  + np.random.normal(0,np.sqrt(number))
    return reward

def Chu(p, q):
    if p == 0:
        return 0
    if q == 0:
        return N
    else:
        return p/q

def Int(x):
    if x<1e-10:
        return 0
    else:
        return math.ceil(x)

def f(p):
    if p == -1:
        return 1
    return 2**p
'''
def f(p):
    if p == 0:
        return 1
    S = [1]
    for i in range(1,p+1):
        S.append(S[i-1] - K + 2 * np.sqrt(S[i-1] * 2**16))
    return int(S[p]) - int(S[p-1])
'''
def UCBScore(y,t,n):
    if t == 0:
        return N
    s = y/t + np.sqrt(2 * np.log(n * np.log(n) * np.log(n) + 1)/t)
    return s

def FindTrueBestArm(i, c, Theta):
    k = np.shape(c)[1]
    B = []
    for arm in range(k):
        B.append(Theta[arm].dot(c[i,arm]))
    return max( (v,i) for i,v in enumerate(B))[1]

def Estimation(c, i, p, n, messages,var_matrix,Y,T,A,R):
    [m,k,d] = np.shape(c)
    estimations = messages[0]
    B = []
    C = []
    for arm in range(k):
      if arm in A[i]:  
        if ((estimations[arm] == (np.zeros([d]) + Max)).all()):
            B.append(UCBScore(Y[i,arm], T[i,arm],1))
            C.append(T[i,arm])
        else:
            B.append(estimations[arm].dot(c[i,arm]))
            C.append(var_matrix[arm].dot(c[i,arm]).dot(c[i,arm]))
      else:
          B.append(0)
          C.append(1)
    return B,C

def FindBestUCBArm(i, n, Y, T):
    k = np.shape(Y)[1]
    B = []
    for arm in range(k):
        B.append(UCBScore(Y[i,arm], T[i,arm], n))
    return max( (v, i) for i, v in enumerate(B) )[1]
            
def PotentialSets(local_potential):
    [m,k] = np.shape(local_potential)
    A = []
    R = []
    for i in range(m):
        A.append([])
        for a in range(k):
            if local_potential[i,a] == 1:
                A[i].append(a)
        if len(A[i]) == 0:
            pdb.set_trace()
    for a in range(k):
        R.append([])
        for i in range(m):
            if a in A[i]:
                R[a].append(i)
    return A,R

def Broadcast(p, loc_info, messages, pi, A, R):
    [m,k,d] = np.shape(loc_info)
    Var_matrix = []
    for arm in range(k):
        X = np.zeros([d,d])
        sum_theta = np.zeros(d)
        for i in R[arm]:
            if ((loc_info[i,arm,:] == (np.zeros([d])+Max)).all()):
                X += np.zeros([d,d])
            else:
                X += Int(pi[i,arm]*f(p-1))*loc_info[i,arm].reshape([d,1]).dot(loc_info[i,arm].reshape([1,d]))/loc_info[i,arm].dot(loc_info[i,arm]) 
                sum_theta += Int(pi[i,arm]*f(p-1))*loc_info[i,arm]
        var_matrix = X
        theta_hat = sum_theta.dot(np.linalg.pinv(var_matrix/f(p-1)))/f(p-1)
        messages[0][arm] = theta_hat
        Var_matrix.append(np.linalg.pinv(var_matrix))
    return messages, Var_matrix

def FindGap(c, Theta):
    [m,k,d] = np.shape(c)
    Gap = np.zeros([m,k])
    deltamin = 1
    deltamax = 0
    for i in range(m):
        mu_star = Theta[FindTrueBestArm(i,c,Theta)].dot(c[i,FindTrueBestArm(i,c,Theta)])
        for arm in range(k):
            Gap[i,arm] = mu_star - Theta[arm].dot(c[i,arm])
            if Gap[i,arm]!=0:
                if Gap[i,arm] < deltamin:
                    deltamin = Gap[i,arm]
                if Gap[i,arm] > deltamax:
                    deltamax = Gap[i,arm]
    return Gap, deltamin, deltamax

def Broadcast_collaborate(c, p, loc_info, messages, pi, A, R):
    [m,k,d] = np.shape(loc_info)
    Var_matrix = []
    for arm in range(k):
        X = np.zeros([d,d])
        sum_theta = np.zeros(d)
        for i in R[arm]:
            if ((loc_info[i,arm,:] == (np.zeros([d])+Max)).all()):
                X += np.zeros([d,d])
            else:
                X += Int(pi[i,arm]*f(p-1))*c[i,arm].reshape([d,1]).dot(c[i,arm].reshape([1,d]))
                sum_theta += Int(pi[i,arm]*f(p-1))*c[i,arm].dot(c[i,arm])*loc_info[i,arm]
        var_matrix = X
        theta_hat = sum_theta.dot(np.linalg.pinv(var_matrix/f(p-1)))/f(p-1)
        messages[0][arm] = theta_hat
        Var_matrix.append(np.linalg.pinv(var_matrix))
    return messages, Var_matrix

