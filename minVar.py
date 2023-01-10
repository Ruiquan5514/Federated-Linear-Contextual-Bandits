import math
import random
import numpy as np
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from random import shuffle
import pdb



def PseudoDeterminent(V):
    eig_values = np.linalg.eig(V)[0]
    det = 1
    for i in eig_values:
        if i > 1e-12:
            det = det*i
    return det

def Gradient(c, pi, A, R):
    [m,k,d] = np.shape(c)
    v = np.zeros([m,k])
    for a in range(k):
        V = np.zeros([d,d])
        for i in R[a]:
            V += pi[i,a]*c[i,a].reshape([d,1]).dot(c[i,a].reshape([1,d]))
        V = np.linalg.pinv(V)
        for i in R[a]:
            v[i,a] = c[i,a].reshape([1,d]).dot(V).dot(c[i,a].reshape([d,1]))
    return v


def Tem(x, tup):
    G = tup[0]
    i = tup[1]
    A = tup[2]
    pi = tup[3]
    s = 0
    index = 0
    for a in A[i]:
        s += -np.log(1 + x[index] * G[i,a])
        index += 1
    return s

def X1(x):
    s = 0
    for i in x:
        s+=i
    return s

def Delta(i, G, pi, A, R):
    x0 = []
    lb = []
    ub = []
    for a in A[i]:
        ub.append(1-pi[i,a])
        x0.append(0.0)
        lb.append(-pi[i,a])
    cons = ({'type': 'eq', 'fun': lambda x:  X1(x)})
    bnds = scipy.optimize.Bounds(lb,ub)
    res = minimize(Tem, x0, [G, i, A, pi], bounds = bnds, constraints = cons)
    return res

def OPTIMIZE(i, c, pi, U, A, R):
    [m,k,d] = np.shape(c)
    G = np.zeros([m,k])
    for client in range(m):
        for a in A[client]:
            G[client,a] = (U[a].dot(c[client,a])).dot(c[client,a])
    delta = Delta(i, G, pi, A, R).x
    index = 0
    for a in A[i]:
        pi[i,a] = pi[i,a] + delta[index]
        tem_matrix = U[a].dot(c[i,a].reshape([d,1]))
        tem_matrix = tem_matrix.dot(c[i,a].reshape([1,d]))
        tem_matrix = tem_matrix.dot(U[a])
        tem_matrix = delta[index] * tem_matrix/(1 + delta[index] * (U[a].dot(c[i,a])).dot(c[i,a]))
        U[a] = U[a] - tem_matrix 
        index += 1
    return pi, U   


def OptimalExperiment(c, A, R, epsilon):
    [m,k,d] = np.shape(c)
    pi = np.zeros([m,k])
    for a in range(k):
        for i in R[a]:
            pi[i,a] = 1.0/len(A[i])
    U = []
    D = 0
    for a in range(k):
        U_a = np.zeros([d,d])
        for i in R[a]:
            U_a += pi[i,a] * c[i,a].reshape([d,1]).dot(c[i,a].reshape([1,d]))
        D += np.linalg.matrix_rank(U_a)
        U.append(np.linalg.pinv(U_a))
    
    for iteration in range(10):
        for i in range(m):
            pi, U = OPTIMIZE(i, c, pi, U, A, R)
        G = np.zeros([m,k])
        for i in range(m):
            for a in A[i]:
                G[i,a] = (U[a].dot(c[i,a])).dot(c[i,a])
        sum_i = np.sum(np.max(G,axis=1))
        if sum_i < D + epsilon:
            break
    print('G(pi) = ', sum_i, ', D = ', D)
    return pi, iteration












