import numpy as np
import pandas as pd
from utils import *

def f_e1(X, wd, Z, beta):
    return sigmoid(wd[0] + wd[1]*X[:,0] + wd[2]*X[:,1] + wd[3]*X[:,2] + wd[4]*X[:,3] + wd[5]*X[:,4]+ beta*Z)

def f_a(X, wa):
    return sigmoid(wa[0] + wa[1]*X[:,0] + wa[2]*X[:,1] + wa[3]*X[:,2])

def f_mu(X, wm):
    return sigmoid(wm[0] + wm[1]*X[:,0] + wm[2]*X[:,1] + wm[3]*X[:,2] + wm[4]*X[:,3] + wm[5]*X[:,4])


def sigmoid_dgp(dgp):

    N = dgp['N']
    Z = np.random.choice(dgp['nz'], N).astype(np.float64)
    X = np.random.normal(0, 1, size=(N,5))

    pD = f_e1(X, dgp['wd'], Z, dgp['beta'])
    pA = f_a(X, dgp['wa'])
    pY = f_mu(X, dgp['w_mu1'])*pD + f_mu(X, dgp['w_mu0'])*(1-pD)

    D = np.random.binomial(1, pD, size=N)
    A = np.random.binomial(1, pA, size=N)
    Y = np.random.binomial(1, pY, size=N)
    
    age = np.zeros_like(pD)
    age[X[:,0]<0] = np.random.binomial(1, .1, size=N)[X[:,0]<0]
    age[X[:,0]>=0] = np.random.binomial(1, .8, size=N)[X[:,0]>=0]

    race = np.zeros_like(pD)
    race[X[:,1]<-.2] = np.random.binomial(1, .7, size=N)[X[:,1] <- .2]
    race[(X[:,1] >= -.2) & (X[:,1] < .2)] = np.random.binomial(1, .8, size=N)[(X[:,1] >= -.2) & (X[:,1] < .2)]
    race[(X[:,1] >= .2)] = np.random.binomial(1, .9, size=N)[(X[:,1] >= .2)]
    

    return {
        'X': X, 
        'Z': Z, 
        'D': D, 
        'Y': Y, 
        'A': A,
        'age': age,
        'race': race
    }

def bernoulli_3d(dgp):

        D = np.random.binomial(1, dgp['pD'], size=dgp['N'])
        A = np.random.binomial(1, dgp['pA'], size=dgp['N'])
        Y = np.random.binomial(1, dgp['pY'], size=dgp['N'])

        RMAG =  np.random.binomial(1, .9, size=dgp['N'])
        DA_corr =  np.random.binomial(1, .5, size=dgp['N'])

        A[(D == 0) & (DA_corr == 1)] = 0
        D[(A == 1) & (D == 0) & (Y == 1) & (RMAG == 1)] = 1 
        A[(A == 1) & (D == 0) & (Y == 1) & (RMAG == 1)] = 0


        vstats = get_v_stats(D, A, Y)

        data = { 
            'D': D, 
            'Y': Y, 
            'A': A,
        }

        return data, vstats