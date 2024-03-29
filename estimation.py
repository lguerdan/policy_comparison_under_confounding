import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

import utils
import vset
import bounds
import dgp as dgp_funcs

def estimate_bounds(dgp, data, id_method, est_method, u=False, K=5):
    '''This function will call the appropriate nuisance estimation function and identification approach'''
    
    if est_method == 'oracle':        
        probs = oracle_nuisance_probs(dgp, data)
        Vpf_down, Vpf_up = vset.get_vset(dgp, data, probs, id_method, est_method)
        bdf = bounds.get_bounds(data, Vpf_down, Vpf_up, u, verbose=False)

    if est_method == 'plugin' or est_method == 'dr':
        bdf = sample_split_crossfit(dgp, data, id_method, est_method, K)
        
    bdf['id_method'] = id_method
    bdf['est_method'] = est_method
    
    return bdf
    
def sample_split_crossfit(dgp, data, id_method, est_method, K, u=False):
    
    in_folds, out_folds = utils.k_fold_split_and_complement(data, K)

    fold_bdfs = []
    
    for k in range(K):

        # Set-up datasets for fold K
        in_data, out_data = in_folds[k], out_folds[k]
        in_dgp, out_dgp = dgp.copy(), dgp.copy()
        in_dgp['N'] = in_data['XU'].shape[0]
        out_dgp['N'] = out_data['XU'].shape[0]

        # Learn models, then run inference on data from fold k
        in_probs = plugin_nuisance_probs(in_dgp, out_dgp, in_data, out_data)
        Vpf_down, Vpf_up = vset.get_vset(in_dgp, in_data, in_probs, id_method, est_method)
        fold_bdfs.append(bounds.get_bounds(data, Vpf_down, Vpf_up, u, verbose=False)) 

    return utils.average_numeric_dataframes(fold_bdfs)

def oracle_nuisance_probs(dgp, data):
        
    XU, D, Y, Z, p_mu1, p_e1, p_mu = data['XU'], data['D'], data['Y'], data['Z'], data['p_mu_1'], data['p_e1'], data['p_mu']
    Dx = dgp['Dx']
    
    # We don't have access to confounders when computing bounds.
    mu1_coeffs = dgp['mu1_coeffs'].copy()
    e1_coeffs = dgp['e1_coeffs'].copy()
    t_coeffs = dgp['t_coeffs'].copy()
    mu1_coeffs[Dx:] = 0
    e1_coeffs[Dx:] = 0
    
    p_mu1_z = np.zeros((dgp['nz'], dgp['N']))
    p_e1_z = np.zeros((dgp['nz'], dgp['N']))

    p_mu1 = dgp_funcs.mu(dgp, mu1_coeffs, XU, Z)
    p_e1 = dgp_funcs.e1(dgp, e1_coeffs, XU, Z)
    p_pi = dgp_funcs.pi(dgp, t_coeffs, XU)

    for z in range(dgp['nz']):
        p_mu1_z[z] = dgp_funcs.mu(dgp, mu1_coeffs, XU, z)
        p_e1_z[z] = dgp_funcs.e1(dgp, e1_coeffs, XU, z)

    return {
        'p_pi': p_pi,
        'p_mu1': p_mu1,
        'p_e1': p_e1,
        'p_mu1_z': p_mu1_z,
        'p_e1_z': p_e1_z
    }

def plugin_nuisance_probs(in_dgp, out_dgp, in_data, out_data):
            
    # Train model via data from held-out folds
    XU, D, Y, Z = out_data['XU'], out_data['D'], out_data['Y'], out_data['Z']

    if 'p_mu_1' in out_data:
        synthetic=True
        p_mu1, p_e1 = out_data['p_mu_1'], out_data['p_e1']
        p_mu1_k, p_e1_k = in_data['p_mu_1'], in_data['p_e1']

        # We know oracle probabilities of the new policy
        t_coeffs = in_dgp['t_coeffs'].copy()
        p_pi = dgp_funcs.pi(in_dgp, t_coeffs, in_data['XU'])


    else: 
        synthetic=False
        # The oracle probabilities are just a deterministic function of T
        p_pi = in_data['T']
    
    # We don't have access to confounders when computing bounds.
    mask = np.ones(XU.shape[1])
    if 'Dx' in out_dgp:
        mask[out_dgp['Dx']:] = 0
    X = XU[:,mask==1].copy()
    XZ = np.concatenate((X, Z.reshape(-1,1)), axis=1)

    if in_dgp['model'] == 'LR':
        mu_hat = LogisticRegression()
        e1_hat = LogisticRegression()

    elif in_dgp['model'] == 'GB':
        mu_hat = GradientBoostingClassifier()
        e1_hat = GradientBoostingClassifier()

    else:
        raise "No model specified for nuisance probs"

    mu_hat.fit(XZ[D==1], Y[D==1])
    e1_hat.fit(XZ, D)
    
    # Regress models on data from fold k
    XU_k, D_k, Y_k, Z_k = in_data['XU'], in_data['D'], in_data['Y'], in_data['Z']
    
    # We don't have access to confounders when computing bounds.
    mask = np.ones(XU_k.shape[1])
    if 'Dx' in in_dgp:
        mask[in_dgp['Dx']:] = 0
    X_k = XU_k[:,mask==1].copy()
    XZ_k = np.concatenate((X_k, Z_k.reshape(-1,1)), axis=1)

    p_mu1_z = np.zeros((in_dgp['nz'], in_dgp['N']))
    p_e1_z = np.zeros((in_dgp['nz'], in_dgp['N']))

    for z in range(in_dgp['nz']):
        Zn = (z*np.ones_like(Z_k)).reshape(-1,1)
        Xz_k = np.concatenate((X_k, Zn), axis=1)
        p_mu1_z[z] = mu_hat.predict_proba(Xz_k)[:,1]
        p_e1_z[z] = e1_hat.predict_proba(Xz_k)[:,1]
        

    p_mu1_hat = mu_hat.predict_proba(XZ_k)[:,1]
    p_e1_hat = e1_hat.predict_proba(XZ_k)[:,1]
    
    if synthetic:
        print('outcome regression error:', (np.abs(p_mu1_hat - p_mu1_k)).mean())
        print('propensitiy error:', (np.abs(p_e1_hat - p_e1_k)).mean())

    return {
        'p_mu1': p_mu1_hat,
        'p_e1': p_e1_hat,
        'p_mu1_z': p_mu1_z,
        'p_e1_z': p_e1_z,
        'p_pi': p_pi
    }
    