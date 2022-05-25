import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb
import minVar
import Functions as F
import SyntheticProblem as Construction
#import MovieLensProblem as Construction
import pandas as pd
import time
import FedUCB

Max = 100000
N = 100000



M = Construction.M
K = Construction.K
D = Construction.D
l = Construction.l
Theta = Construction.Theta
c = Construction.c

def Fed_PE(c, Theta, horizon):
    print('Fed-PE begins: ')
    [m,k,d] = np.shape(c)
    
    # alpha
    alpha = 2 * np.log(m * k * horizon * 10 * 2)
    k_0 = 1
    while k_0 * d < 2 * np.log(k * horizon * 10) + d * np.log(k_0) + d:
        k_0 += 0.1
    alpha = min(alpha, 2 * np.log(k * horizon * 10) + d * np.log(k_0) + d)
    
    # Gap
    Gap,deltamin,deltamax = F.FindGap(c, Theta)
    
    # Storage vectors and matrix
    E = np.zeros([m, k, d])
    Y = np.zeros([m, k])
    T = np.zeros([m, k])
    mu1 = np.zeros([m, k])
    mu2 = np.zeros([m, k])
    messages = [np.zeros([k, d]) + Max, np.zeros([k, 2])]
    x_axis = [0]
    y_axis = [0]
    time = 0
    local_potential = np.zeros([m, k]) + 1
    var_matrix = 0
    pi = np.zeros([m, k])
    regret = 0
    n = 0
    
    # Initialization
    local_information = np.zeros([m, k, d]) + Max
    for i in range(m):
        for a in range(k):
            Y[i,a] += F.PullArm(a, c[i,a], Theta, 1)
            T[i,1] += 1
            local_information[i,a] = c[i,a] * Y[i,a] / (c[i,a].dot(c[i,a]))
            E[i,a] = local_information[i,a] / np.sqrt(local_information[i,a].dot(local_information[i,a]))
    A, R = F.PotentialSets(local_potential)
    messages, var_matrix = F.Broadcast(1, local_information, messages, pi+1, A, R)

    for p in range(1, horizon):
        print('phase', p)
        local_information = np.zeros([m, k, d]) + Max
        A, R = F.PotentialSets(local_potential)
        pi0 = np.zeros([m, k])
        
        # Arm Elimination
        for i in range(m):
            B,C = F.Estimation(c, i, p, n, messages, var_matrix, Y, T, A, R)
            b = max( (v, b) for b, v in enumerate(B) )[1]
            pi0[i,b] = 1
            mu1[i,b] = B[b]*C[b]
            mu2[i,b] = C[b]
            for a in range(k):
                if local_potential[i,a] == 1:
                    mu1[i,a] = B[a]*C[a]
                    mu2[i,a] = C[a]
                    if B[b] - B[a] > np.sqrt(alpha * mu2[i,b] / l) + np.sqrt(alpha * mu2[i,a] / l):
                        local_potential[i, a] = 0

            if local_potential[i].dot(local_potential[i]) == 0:
                print('wrong elimination, all arms deleted.')
                pdb.set_trace()
        
        # Server: Optimization Problem
        A, R = F.PotentialSets(local_potential)
        pi, iterations = minVar.OptimalExperiment(E, A, R, 0.1)

        # Clients: Exploration      
        for i in range(m):    
            for a in A[i]:
                if local_potential[i,a] == 1:
                    phase_reward = 0
                    number = F.Int(F.f(p) * pi[i,a])
                    for pull_number in range(number):
                        instance_reward = F.PullArm(a, c[i,a], Theta)
                        phase_reward += instance_reward
                        Y[i,a] += instance_reward
                        T[i,a] += 1
                        regret += Gap[i,a]
                    if number > 0:
                        local_information[i,a] = c[i,a] * phase_reward / number / (c[i,a].dot(c[i,a]))
        
        # Server: Aggregation   
        messages, var_matrix = F.Broadcast(p+1, local_information, messages, pi, A, R)
         
        n=0    
        for a in range(k):
            n += T[0,a]
        x_axis.append(n)
        y_axis.append(regret/m)
    '''
    print(Gap,deltamin,deltamax)
    print('whether delete optimal arm: ',np.trace(local_potential.dot(Gap.transpose())))
    print('whether delete optimal arm: ',np.trace(local_potential.dot(local_potential.transpose())))
    print('regret per client: ', regret/m)
    '''
    return x_axis, y_axis
'''
horizon = 16
n = 1
S1 = np.zeros(horizon)
for i in range(n):
    x_axis, S = Fed_PE(c, Theta, horizon) 
    S1 = S1 + np.array(S)
S1 = S1/(n + 0.0)
plt.plot(np.array(x_axis), S1, label = 'Fed-PE')
'''


def UCB(c, Theta, horizon):
    print('Local UCB begins: ')
    [m, k, d] = np.shape(c)
    
    # Gap
    Gap,deltamin,deltamax = F.FindGap(c, Theta)
    Y = np.zeros([m, k])
    T = np.zeros([m, k])
    x_axis = [0]
    y_axis = [0]
    regret = 0

    for n in range(horizon):
        if n % 10000 == 0: print('T = ', n) 
        for i in range(m):
                a_i = F.FindBestUCBArm(i, n+1, Y, T)
                Y[i,a_i] += F.PullArm(a_i, c[i,a_i], Theta)
                T[i,a_i] += 1
                regret += Gap[i, a_i]
        x_axis.append(n+1)
        y_axis.append(regret/m) 
      
    print('regret per client: ', regret/m)   
    return x_axis, y_axis

'''
n=1
horizon = 2**16
S1 = np.zeros(horizon+1)
for i in range(n):
    x_axis, S = UCB(c,Theta,horizon)  
    S1 = S1 + np.array(S)
S1 = S1/(n + 0.0)
plt.plot(np.array(x_axis), S1, label = 'local UCB')
'''




def Enhanced(c, Theta, horizon):
    print('Enhanced Fed-PE begins: ')
    [m, k, d] = np.shape(c)

    # alpha
    alpha = m * (m * k * 10)**2 / k / d

    # Gap
    Gap,deltamin,deltamax = F.FindGap(c, Theta)

    # Storage vectors and matrices
    E = np.zeros([m, k, d])
    Y = np.zeros([m, k])
    T = np.zeros([m, k])
    mu1 = np.zeros([m, k])
    mu2 = np.zeros([m, k]) + d * k / m
    N = 0
    messages = [np.zeros([k, d]) + Max, np.zeros([k, 2])]
    
    x_axis = [0]
    y_axis = [0]
    Time = [0]
    sparse = [m * k]
    Iteration = [0]
    
    local_potential = np.zeros([m, k]) + 1
    var_matrix = 0
    pi = np.zeros([m, k])
    regret = 0
    n = 0
    
    # Initialization
    local_information = np.zeros([m, k, d]) + Max
    for i in range(m):
        for a in range(k):
            Y[i,a] += F.PullArm(a, c[i,a], Theta, 1)
            T[i,1] += 1
            local_information[i,a] = c[i,a] * Y[i,a] / (c[i,a].dot(c[i,a]))
            E[i,a] = local_information[i,a] / np.sqrt(local_information[i,a].dot(local_information[i,a]))
    A, R = F.PotentialSets(local_potential)
    messages, var_matrix = F.Broadcast(1, local_information, messages, pi+1, A, R)
    
    # Clients and Server
    for p in range(1,horizon):
        print('phase: ', p)
        time_start = time.time()
        local_information = np.zeros([m,k,d]) + Max
        A, R = F.PotentialSets(local_potential)
        pi0 = np.zeros([m,k])
        
        # Arm Elimination
        N += F.f(p-1)
        for i in range(m):
            B,C = F.Estimation(c, i, p, n, messages, var_matrix, Y, T, A, R)
            for a in range(k):
                if local_potential[i,a] == 1:
                    mu1[i,a] += B[a] * F.f(p-1)
                    mu2[i,a] += C[a] * F.f(p-1)**2 / l
            B = []
            for a in range(k):
                B.append(mu1[i,a] / N)
            b = max( (v, b) for b, v in enumerate(B) )[1]
            pi0[i,b] = 1
            for a in range(k):
                if local_potential[i,a] == 1:
                    if mu1[i,b] / N - mu1[i,a] / N > np.sqrt(np.log(alpha * mu2[i,b]) * mu2[i,b]) / N + np.sqrt(np.log(alpha * mu2[i,a]) * mu2[i,a]) / N:
                        local_potential[i,a] = 0
            if local_potential[i].dot(local_potential[i]) == 0:
                print('wrong elimination, all arms deleted.')
                pdb.set_trace()
        
        # Server: Optimization Problem
        A, R = F.PotentialSets(local_potential)
        pi, iteration = minVar.OptimalExperiment(E, A, R, 0.1)
        
        # Clients: Exploration
        sparse_p = 0
        for i in range(m):    
            for a in A[i]:
                if local_potential[i,a] == 1:
                    phase_reward = 0
                    number = F.Int(F.f(p) * pi[i,a])
                    for pull_number in range(number):
                        instance_reward = F.PullArm(a, c[i,a], Theta)
                        phase_reward += instance_reward
                        Y[i,a] += instance_reward
                        T[i,a] += 1
                        regret += Gap[i,a]
                    if number > 0:
                        sparse_p += 1
                        local_information[i,a] = c[i,a] * phase_reward / number / (c[i,a].dot(c[i,a]))
        
        #Server: Aggregates
        messages, var_matrix = F.Broadcast(p+1, local_information, messages, pi, A, R)
        
        #Store regret, time, sparsity, iteration
        n=0    
        for a in range(k):
            n += T[0,a]
        x_axis.append(n)
        y_axis.append(regret/m)
        time_end = time.time()
        Time.append(time_end - time_start)
        sparse.append(sparse_p)
        Iteration.append(iteration)
    '''
    print(Gap,deltamin,deltamax)  
    print('whether delete optimal arm: ',np.trace(local_potential.dot(Gap.transpose())))
    print('whether delete optimal arm: ',np.trace(local_potential.dot(local_potential.transpose())))
    print('regret per client: ', regret/m)
    '''
    return x_axis, y_axis, Time, sparse, Iteration

'''
horizon = 16
n = 1
S1 = np.zeros(horizon)
for i in range(n):
    x_axis, Regret, Time, Sparse, Iteration = Enhanced(c, Theta, horizon)  
    S1 = S1 + np.array(Regret)
S1 = S1/(n + 0.0)
plt.plot(np.array(x_axis), S1, label = 'Enhanced Fed-PE')
'''


def Collaborate(c, Theta, horizon):
    print('Collaborative algorithm begins:')
    [m, k, d] = np.shape(c)
    #Pertabation of feature vectors
    c_hat = np.zeros([m,k,d])
    for i in range(m):
        for arm in range(k):
            for s in range(d):
                c_hat[i,arm,s] = c[i,arm,s] + np.random.normal(0,1)

    alpha = m * (m * k * 10)**2 / k / d
    Gap,deltamin,deltamax = F.FindGap(c,Theta)
    Y = np.zeros([m, k])
    T = np.zeros([m, k])
    mu1 = np.zeros([m, k])
    mu2 = np.zeros([m, k]) + d * k / m
    N = 0
    messages = [np.zeros([k, d]) + Max, np.zeros([k, 2])]
    x_axis = [0]
    y_axis = [0]
    time = 0
    local_potential = np.zeros([m, k]) + 1
    var_matrix = 0
    pi = np.zeros([m, k])
    regret = 0
    n = 0

    # Initialization
    local_information = np.zeros([m, k, d]) + Max
    for i in range(m):
        for a in range(k):
            Y[i,a] += F.PullArm(a, c[i,a], Theta, 1)
            T[i,1] += 1
            local_information[i,a] = c_hat[i,a] * Y[i,a] / (c_hat[i,a].dot(c_hat[i,a]))
    A, R = F.PotentialSets(local_potential)
    messages, var_matrix = F.Broadcast_collaborate(c, 1, local_information, messages, pi+1, A, R)

    for p in range(1, horizon):
        print('phase: ', p)
        local_information = np.zeros([m, k, d]) + Max
        A, R = F.PotentialSets(local_potential)
        pi0 = np.zeros([m, k])
        N += F.f(p-1)
        
        # Clients: Arm Elimination
        for i in range(m):
            B,C = F.Estimation(c, i, p, n, messages, var_matrix, Y, T, A,  R)
            for a in range(k):
                if local_potential[i,a] == 1:
                    mu1[i,a] += B[a]*F.f(p-1)
                    mu2[i,a] += C[a]*F.f(p-1)**2
            
            B = []
            for a in range(k):
                B.append(mu1[i,a] / N)
            b = max( (v, b) for b, v in enumerate(B) )[1]
            pi0[i,b] = 1
            for a in range(k):
                if local_potential[i,a] == 1:
                    if mu1[i,b]/N - mu1[i,a]/N > np.sqrt(np.log(alpha * mu2[i,b]) * mu2[i,b]) / N + np.sqrt(np.log(alpha * mu2[i,a]) * mu2[i,a]) / N:
                        local_potential[i,a] = 0
            if local_potential[i].dot(local_potential[i]) == 0:
                print('wrong elimination, all arms deleted.')
                pdb.set_trace()
        
        # Server: Optimization Problem
        A, R = F.PotentialSets(local_potential)
        pi, iterations = minVar.OptimalExperiment(c_hat, A, R, 1.0/10)
        
        # Clients: Exploration    
        for i in range(m):    
            for a in A[i]:
                if local_potential[i,a] == 1:
                    phase_reward = 0
                    number = F.Int(F.f(p) * pi[i,a])
                    for pull_number in range(number):
                        instance_reward = F.PullArm(a, c[i,a], Theta)
                        phase_reward += instance_reward
                        Y[i,a] += instance_reward
                        T[i,a] += 1
                        regret += Gap[i,a]
                    if number > 0:
                        local_information[i,a] = c_hat[i,a] * phase_reward / number / (c_hat[i,a].dot(c_hat[i,a]))
        
        # Server: Aggregation 
        messages, var_matrix = F.Broadcast_collaborate(c_hat, p+1, local_information, messages, pi, A, R)
        
        n=0    
        for a in range(k):
            n += T[0,a]
        x_axis.append(n)
        y_axis.append(regret/m)
    print(Gap,deltamin,deltamax)  
    print('whether delete optimal arm: ',np.trace(local_potential.dot(Gap.transpose())))
    print('whether delete optimal arm: ',np.trace(local_potential.dot(local_potential.transpose())))
    print('regret per client: ', regret/m)
    return x_axis, y_axis


horizon = 14
n = 1
S1 = np.zeros(horizon)
for i in range(n):
    x_axis, S = Collaborate(c, Theta, horizon)
    S1 = S1 + np.array(S)
S1 = S1/(n + 0.0)
plt.plot(np.array(x_axis), S1, label = 'Collaborative')


horizon = 13
n = 1
#S1 = np.zeros(horizon)
for i in range(n):
    x_axis, S = FedUCB.FedUCB(c, Theta, horizon)
    #S1 = S1 + np.array(S)
#S1 = S1/(n + 0.0) 
plt.plot(np.array(x_axis), S, label = 'FedUCB')





plt.xlabel('Time: T')
plt.ylabel('Regret: R(T)')
plt.legend()
plt.grid(ls = '--')
plt.show()  

