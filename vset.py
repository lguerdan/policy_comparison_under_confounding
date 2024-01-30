from dgp import *
import numpy as np


def get_vset(dgp, data, probs, id_method, est_method):

    if id_method == 'NA':
        return compute_na_bounds(dgp, data, probs)

    elif id_method == 'IV': 
        return compute_iv_bounds(dgp, data, probs)

    elif id_method == 'MSM': 
        return compute_msm_bounds(dgp, data, probs, est_method)


def compute_msm_bounds(dgp, data, nuisance_probs, est_method):
    
    XU, T, D, Y = data['XU'], data['T'], data['D'], data['Y']
    p_mu1 = nuisance_probs['p_mu1']
    p_e1 = nuisance_probs['p_e1']
    p_pi = nuisance_probs['p_pi']

    mu_down = np.clip((1/dgp['lambda']) * p_mu1, 0, 1)
    mu_up = np.clip(dgp['lambda'] * p_mu1, 0, 1)
    p_e0 = 1-p_e1

    if est_method == 'plugin' or est_method == 'oracle':
        v110_up = (T * mu_up * p_e0).mean()
        v100_up = ((1-T) * mu_up * p_e0).mean()
        v110_down = (T * mu_down * p_e0).mean()
        v100_down = ((1-T) * mu_down * p_e0).mean()

    elif est_method == 'dr':
        v110_up = dgp['lambda'] * compute_dr_vstatistic(D, T, Y, p_e0, p_pi, p_mu1, t=1)
        v100_up = dgp['lambda'] * compute_dr_vstatistic(D, T, Y, p_e0, p_pi, p_mu1, t=0)
        v110_down = (1/dgp['lambda']) * compute_dr_vstatistic(D, T, Y, p_e0, p_pi, p_mu1, t=1)
        v100_down = (1/dgp['lambda']) * compute_dr_vstatistic(D, T, Y, p_e0, p_pi, p_mu1, t=0)
        
    Vpf_down, Vpf_up = np.zeros((2,2)), np.zeros((2,2))
    
    Vpf_down[0,1], Vpf_down[1,1] = v110_down, v110_down
    Vpf_down[0,0], Vpf_down[1,0] = v100_down, v100_down
    Vpf_up[0,1], Vpf_up[1,1] = v110_up, v110_up
    Vpf_up[0,0], Vpf_up[1,0] = v100_up, v100_up

    return Vpf_down, Vpf_up


def compute_dr_vstatistic(D, T, Y, e, p_pi, mu, t):
    
    # Our target is conditional on D=0
    e = 1-e
    
    # Estimate our target conditional on T=t
    if t==0:
        T = 1-T
        p_pi=1-p_pi
    
    return (D*Y*p_pi + (T-p_pi)*e*mu).mean()


def compute_na_bounds(dgp, data, nuisance_probs):
    
    Y, D, T = data['Y'], data['D'], data['T']

    Vpf_down, Vpf_up = np.zeros((2,2)), np.zeros((2,2))
    
    Vpf_up[0,1] = ((D==0) & (T==1)).mean()
    Vpf_up[1,1] = ((D==0) & (T==1)).mean()
    Vpf_up[0,0] = ((D==0) & (T==0)).mean()
    Vpf_up[1,0] = ((D==0) & (T==0)).mean()
    
    return Vpf_down, Vpf_up
 

def compute_iv_bounds(dgp, data, nuisance_probs):
    
    XU, Z, T = data['XU'], data['Z'], data['T']
    N, Dx, Du = dgp['N'], dgp['Dx'], dgp['Du']

    p_mu1 = nuisance_probs['p_mu1']
    p_e1 = nuisance_probs['p_e1']
    p_mu1_z = nuisance_probs['p_mu1_z']
    p_e1_z = nuisance_probs['p_e1_z']

    mu_down_z = np.zeros((dgp['nz'], dgp['N']))
    mu_up_z = np.zeros((dgp['nz'], dgp['N']))
    
    # Compute upper and lower bounds on mu(a,x)
    for z in range(dgp['nz']):
        e0_z = 1-p_e1_z[z]
        mu_down_z[z] = p_e1_z[z] * p_mu1_z[z]
        mu_up_z[z] = e0_z + p_e1_z[z] * p_mu1_z[z]

    mu_down = mu_down_z.max(axis=0)
    mu_up = mu_up_z.min(axis=0)
    
    v110_up = (T * (mu_up - p_mu1 * p_e1)).mean()
    v100_up = ((1-T) * (mu_up - p_mu1 * p_e1)).mean()

    v110_down = (T * (mu_down - p_mu1 * p_e1)).mean()
    v100_down = ((1-T) * (mu_down - p_mu1 * p_e1)).mean()
    
    Vpf_down = np.zeros((2,2))
    Vpf_up = np.zeros((2,2))
    
    Vpf_down[0,1], Vpf_down[1,1] = v110_down, v110_down
    Vpf_down[0,0], Vpf_down[1,0] = v100_down, v100_down
    Vpf_up[0,1], Vpf_up[1,1] = v110_up, v110_up
    Vpf_up[0,0], Vpf_up[1,0] = v100_up, v100_up

    return Vpf_down, Vpf_up
