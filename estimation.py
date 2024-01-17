import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import utils
import vset
import bounds
import dgp as dgp_funcs

def estimate_bounds(dgp, data, id_method, est_method, K=5):
    '''This function will call the appropriate nuisance estimation function and identification approach'''
    
    if est_method == 'oracle':        
        probs = oracle_nuisance_probs(dgp, data)
        Vpf_down, Vpf_up = vset.get_vset(dgp, data, probs, id_method)
        bdf = bounds.get_bounds(data, Vpf_down, Vpf_up, verbose=False)

    if est_method == 'plugin':
        bdf = sample_split_crossfit(dgp, data, id_method, est_method, K)
        
    bdf['id_method'] = id_method
    bdf['est_method'] = est_method
    
    return bdf
    
def sample_split_crossfit(dgp, data, id_method, est_method, K):
    
    in_folds, out_folds = utils.k_fold_split_and_complement(data, K)

    fold_bdfs = []
    
    for k in range(K):

        # Set-up datasets for fold K
        in_data, out_data = in_folds[k], out_folds[k]
        in_dgp, out_dgp = dgp.copy(), dgp.copy()
        in_dgp['N'] = in_data['XU'].shape[0]
        out_dgp['N'] = out_data['XU'].shape[0]

        # Learn models, then run inference via data from fold k
        in_probs = plugin_nuisance_probs(in_dgp, out_dgp, in_data, out_data)
        Vpf_down, Vpf_up = vset.get_vset(in_dgp, in_data, in_probs, id_method)
        fold_bdfs.append(bounds.get_bounds(data, Vpf_down, Vpf_up, verbose=False)) 

    return utils.average_numeric_dataframes(fold_bdfs)

def oracle_nuisance_probs(dgp, data):
        
    XU, D, Y, Z, p_mu1, p_e1, p_mu = data['XU'], data['D'], data['Y'], data['Z'], data['p_mu_1'], data['p_e1'], data['p_mu']
    Dx = dgp['Dx']
    
    # We don't have access to confounders when computing bounds.
    mu1_coeffs = dgp['mu1_coeffs'].copy()
    e1_coeffs = dgp['e1_coeffs'].copy()
    mu1_coeffs[Dx:] = 0
    e1_coeffs[Dx:] = 0
    
    p_mu1_z = np.zeros((dgp['nz'], dgp['N']))
    p_e1_z = np.zeros((dgp['nz'], dgp['N']))

    p_mu1 = dgp_funcs.mu(dgp, mu1_coeffs, XU, Z)
    p_e1 = dgp_funcs.e1(dgp, e1_coeffs, XU, Z)

    for z in range(dgp['nz']):
        p_mu1_z[z] = dgp_funcs.mu(dgp, mu1_coeffs, XU, z)
        p_e1_z[z] = dgp_funcs.e1(dgp, e1_coeffs, XU, z)

    return {
        'p_mu1': p_mu1,
        'p_e1': p_e1,
        'p_mu1_z': p_mu1_z,
        'p_e1_z': p_e1_z
    }

def plugin_nuisance_probs(in_dgp, out_dgp, in_data, out_data):
            
    # Train model via data from held-out folds
    XU, D, Y, Z, p_mu1, p_e1 = out_data['XU'], out_data['D'], out_data['Y'], out_data['Z'], out_data['p_mu_1'], out_data['p_e1']
    
    # We don't have access to confounders when computing bounds.
    mask = np.ones(XU.shape[1])
    mask[out_dgp['Dx']:] = 0
    X = XU[:,mask==1].copy()
    XZ = np.concatenate((X, Z.reshape(-1,1)), axis=1)

    mu_hat = LogisticRegression()
    e1_hat = LogisticRegression()

    mu_hat.fit(XZ[D==1], Y[D==1])
    e1_hat.fit(XZ, D)
    
    # Regress models on data from fold k
    XU_k, D_k, Y_k, Z_k, p_mu1_k, p_e1_k = in_data['XU'], in_data['D'], in_data['Y'], in_data['Z'], in_data['p_mu_1'], in_data['p_e1']
    
    # We don't have access to confounders when computing bounds.
    mask = np.ones(XU_k.shape[1])
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
    
    print('outcome regression error:', (np.abs(p_mu1_hat - p_mu1_k)).mean())
    print('propensitiy error:', (np.abs(p_e1_hat - p_e1_k)).mean())

    return {
        'p_mu1': p_mu1_hat,
        'p_e1': p_e1_hat,
        'p_mu1_z': p_mu1_z,
        'p_e1_z': p_e1_z
    }
    

