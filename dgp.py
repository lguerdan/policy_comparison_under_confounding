import numpy as np
import pandas as pd
from utils import *


def f_a(X, wa):
    return sigmoid(wa[0] + wa[1]*X[:,0] + wa[2]*X[:,1] + wa[3]*X[:,2])

def e1(dgp, coeffs, XU, Z):
    norm = 1/(2*np.sqrt(XU.shape[1]))
    return sigmoid(.18 * (( dgp['e1_coeffs'] * XU).sum(axis=1) + dgp['beta_zd'] * Z))

def mu(dgp, coeffs, XU, Z):
    norm = 1/(2*np.sqrt(XU.shape[1]))
    return sigmoid(.18 * ((coeffs * XU).sum(axis=1) + dgp['beta_zy'] * Z))


def generate_data(dgp):
    
    check_dgp_config(dgp)
    
    # Co-variate information
    N, Dx, Du = dgp['N'], dgp['Dx'], dgp['Du']
    nD = Dx+Du
    
    # Proxy information
    nz, nw, beta_zd, beta_zy = dgp['nz'], dgp['nw'], dgp['beta_zd'], dgp['beta_zy']
    
    e1_coeffs = dgp['e1_coeffs'].copy()
    z_coeffs = dgp['z_coeffs'].copy()
    mu1_coeffs = dgp['mu1_coeffs'].copy()
    mu0_coeffs = dgp['mu0_coeffs'].copy()
    
    norm = 1/(2*np.sqrt(nD))
    T = np.random.binomial(1,.35*np.ones(N))
    
    # Sample measured and unmeasured confounders
    mean, cov = np.zeros(nD), np.eye(nD)
    XU = np.random.multivariate_normal(mean, cov, N)

    # Compute the probability distribution for Z
    prob_Z = np.exp(z_coeffs*XU)
    prob_Z = prob_Z / np.sum(prob_Z.sum(axis=1))  
    weights = np.random.rand(nD, nz)
    logits = np.dot(XU, weights)
    pZ = softmax(logits)

    # Sample instrument values
    Z = np.argmax(np.array([np.random.multinomial(1, p) for p in pZ]), axis=1)

    # Treatment propensity is a function of X, U and Z. Beta is a scaling factor.    
    pD = e1(dgp, e1_coeffs, XU, Z)
    D = np.random.binomial(1, pD)
    p_mu_1 = mu(dgp, mu1_coeffs, XU, Z)

    if dgp['id_assumption'] == 'MSM':
        p_mu_0 = np.clip(dgp['lambda_star'] * p_mu_1, 0, 1)
    else:
        p_mu_0 = mu(dgp, dgp['mu0_coeffs'], XU, Z)
    
    p_mu = pD*p_mu_1 + (1-pD)*p_mu_0
    Y = np.random.binomial(1, p_mu)

    return {
        'Y': Y,
        'pY': p_mu,
        'pD': pD,
        'D': D,
        'XU': XU,
        'Z': Z,
        'T': T
    }


def check_dgp_config(dgp):
    
    if dgp['id_assumption'] != 'IV':
        assert dgp['beta_zd'] == 0, 'Error in configuration, beta_zd loading should be zero in non-iv setting'
        assert dgp['beta_zy'] == 0, 'Error in configuration, beta_zy loading should be zero in non-iv setting'
        
    if dgp['id_assumption'] == 'IV':
        assert dgp['z_coeffs'][Dx:].sum() == 0, 'Error in configuration, instrument is confounded'
        
    if dgp['id_assumption'] == 'MSM':
        assert dgp['z_coeffs'].sum() == 0, 'Error in configuration, instrument available in MSM setting'

