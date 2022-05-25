import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb
import Functions as F




def FedUCB(c, Theta, horizon):
    print('FedUCB begins')
    [m,k,d] = np.shape(c)
    alpha = 0.1
    V_matrix = np.zeros([m,k,d,d])
    v_vector = np.zeros([m,k,d])

    S_matrix = np.zeros([m,k,d,d])
    s_vector = np.zeros([m,k,d])
    U_matrix = np.zeros([m,k,d,d])
    u_vector = np.zeros([m,k,d])

    theta_hat = np.zeros([m,k,d])

    Gap,deltamin,deltamax = F.FindGap(c, Theta)
    x_axis = [0]
    R = [0]
    regret = 0
    delta_t = 1e-6
    syn = 0
    Lam = 5

    for i in range(m):
        for arm in range(k):
            for s in range(d):
                S_matrix[i,arm,s,s] = Lam

    print(c)
    print(Theta)
    for t in range(2**horizon):
        V_matrix = S_matrix + U_matrix
        v_vector = s_vector + u_vector
        for i in range(m):
            B = []
            beta = np.sqrt(Lam) + np.sqrt(2*np.log(2/alpha) + d * np.log(np.linalg.det(V_matrix[i,arm])/Lam))
            for arm in range(k):
                
                v_inv = np.linalg.inv(V_matrix[i,arm])

                theta_hat[i,arm] = np.dot(v_inv, v_vector[i,arm])
                #pdb.set_trace()
            
                B.append(c[i,arm].dot(theta_hat[i,arm]) + beta*np.sqrt(c[i,arm].dot(v_inv.dot(c[i,arm]))))
            
            b = max( (v, b) for b, v in enumerate(B) )[1]
            instance_reward = F.PullArm(b, c[i,b], Theta)
            regret += Gap[i,b]
            U_matrix[i,b] += (c[i,b].reshape([d,1])).dot(c[i,b].reshape([1,d]))
            u_vector[i,b] += instance_reward * c[i,b]
            delta_t += 1
            
            if np.log(np.linalg.det(V_matrix[i,b] + (c[i,b].reshape([d,1])).dot(c[i,b].reshape([1,d]))) \
                / np.linalg.det(S_matrix[i,b]) ) > 2 * 2**horizon * d * ( 1 + np.log(1 + 2**horizon / d / Lam) )/((horizon/2-2)**2)/delta_t:
                syn = 1
        
        if syn == 1:
            print('syn at t=',t)
            x_axis.append(t)
            R.append(regret/m)
            S_middle = np.zeros([k,d,d])
            s_middle = np.zeros([k,d])
            for arm in range(k):
                for i in range(m):
                    S_middle[arm] += U_matrix[i,arm]
                    s_middle[arm] += u_vector[i,arm]
            U_matrix = np.zeros([m,k,d,d])
            u_vector = np.zeros([m,k,d])
                    
            delta_t = 1e-6
            syn = 0
            for i in range(m):
                for arm in range(k):
                    S_matrix[i,arm] += S_middle[arm]
                    s_vector[i,arm] += s_middle[arm]

        else:
            delta_t += 1
    
    pdb.set_trace()
    return x_axis, R





            
            



    




