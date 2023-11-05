from dgp import *
import pandas as pd
import numpy as np


def assert_ubs_coverage(data, ubs, eps=0.01):
    '''Assert coverage of estimated bounds on unobserved quantities'''

    A, D, Y = data['A'], data['D'], data['Y']
    
    # Bounds on unidentified terms
    v10_down, v10_up = ubs['v10_down'], ubs['v10_up']
    v00_down, v00_up = ubs['v00_down'], ubs['v00_up']
    w10_down, w10_up = ubs['w10_down'], ubs['w10_up']
    w00_down, w00_up = ubs['w00_down'], ubs['w00_up']

    v10_true = ((A==1) & (D==0) & (Y==1)).mean()
    v00_true = ((A==0) & (D==0) & (Y==1)).mean()
    w10_true = ((A==1) & (D==0) & (Y==0)).mean()
    w00_true = ((A==0) & (D==0) & (Y==0)).mean()
    
    # Assert that true ground-truth terms are covered within estimated interval
    assert v10_down-eps <=  v10_true <= v10_up+eps, f'v10: {v10_true:.2}. IV bound [{v10_down:.2}, {v10_up:.2}]'
    assert v00_down-eps <=  v00_true <= v00_up+eps, f'v00: {v00_true:.2}. IV bound [{v00_down:.2}, {v00_up:.2}]'
    assert w00_down-eps <=  w00_true <= w00_up+eps, f'w00: {w00_true:.2}. IV bound [{w00_down:.2}, {w00_up:.2}]'
    assert w10_down-eps <=  w10_true <= w10_up+eps, f'w10: {w10_true:.2}. IV bound [{w10_down:.2}, {w10_up:.2}]'


def one_step_bounds(data, dgp, u, metric, id_strategy):
    
    A, D, Y = data['A'], data['D'], data['Y']
    ubs = unobs_quadrant_bounds(data, dgp, id_strategy)
#     assert_ubs_coverage(data, ubs)
    
    # Bounds on unidentified terms
    v10_down, v10_up = ubs['v10_down'], ubs['v10_up']
    v00_down, v00_up = ubs['v00_down'], ubs['v00_up']
    w10_down, w10_up = ubs['w10_down'], ubs['w10_up']
    w00_down, w00_up = ubs['w00_down'], ubs['w00_up']

    # Empirical estimates of identified terms
    v11 = ((D==1) & (A==1) & (Y==1)).mean()
    v01 = ((D==1) & (A==0) & (Y==1)).mean()
    w11 = ((D==1) & (A==1) & (Y==0)).mean()
    w01 = ((D==1) & (A==0) & (Y==0)).mean()
    
    if metric=='FNR':
        d = v00_up if v01 > v10_up else v00_down
        R_down = (v01 - v10_up)/ (d + v10_up + v01 + v11)
        
        u = v00_down if v01 > v10_down else v00_up
        R_up = (v01 - v10_down)/ (u + v10_down + v01 + v11)
        
    elif metric=='TNR':
        d = w00_up if w01 > w10_up else w00_down
        R_down = (w01 - w10_up)/(d+w10_up+w01+w11)
        
        c = w00_down if w01 > w10_down else w00_up
        R_up = (w01 - w10_down)/(c+w10_down+w01+w11)
        
    elif metric=='TPR':
        d = v00_up if v10_down > v01 else v00_down
        R_down = (v10_down - v01)/ (d + v10_down + v01 + v11)
        
        c = v00_down if v10_up > v01 else v00_up
        R_up = (v10_up - v01)/ (c + v10_up + v01 + v11)
        
    elif metric=='FPR':
        d = w00_up if w10_down > w01 else w00_down
        R_down = (w10_down-w01)/(d+w10_down+w01+w11)
        
        u = w00_down if w10_up > w01 else w00_up
        R_up = (w10_up-w01)/(u+w10_up+w01+w11)
        
    elif metric=='ACCURACY':
        R_down = w01 + v10_down - w10_up - v01
        R_up = w01 + v10_up - w10_down - v01
        
    elif metric=='PV':
        R_down = u[0,0]*(w01-w10_up) + u[0,1]*(v01-v10_up) + u[1,0]*(w10_down-w01)+u[1,1]*(v10_down-v01)
        R_up = u[0,0]*(w01-w10_down) + u[0,1]*(v01-v10_down) + u[1,0]*(w10_up-w01)+u[1,1]*(v10_up-v01)
        
    elif metric=='PV_cost':
        R_down = u[0,0]*(w01-w10_down) + u[0,1]*(v01-v10_down) + u[1,0]*(w10_up-w01)+u[1,1]*(v10_up-v01)
        R_up = u[0,0]*(w01-w10_up) + u[0,1]*(v01-v10_up) + u[1,0]*(w10_down-w01)+u[1,1]*(v10_down-v01)
        
    return R_down, R_up

def two_step_bounds(data, dgp, u, metric, id_strategy):
    
    A, D, Y = data['A'], data['D'], data['Y']
    ubs = unobs_quadrant_bounds(data, dgp, id_strategy)
#     assert_ubs_coverage(data, ubs)
    
    # Bounds on unidentified terms
    v10_down, v10_up = ubs['v10_down'], ubs['v10_up']
    v00_down, v00_up = ubs['v00_down'], ubs['v00_up']
    w10_down, w10_up = ubs['w10_down'], ubs['w10_up']
    w00_down, w00_up = ubs['w00_down'], ubs['w00_up']
    
    # Empirical estimates of identified terms
    v11 = ((D==1) & (A==1) & (Y==1)).mean()
    v01 = ((D==1) & (A==0) & (Y==1)).mean()
    w11 = ((D==1) & (A==1) & (Y==0)).mean()
    w01 = ((D==1) & (A==0) & (Y==0)).mean()
    
    if metric=='FNR':
    
        TS_down = (v00_down + v01)/(v00_down + v10_up + v01 + v11) - \
                (v00_up + v10_up)/(v00_up + v10_up + v01 + v11)

        TS_up = (v00_up + v01) / (v00_up + v10_down + v01 + v11) - \
                (v00_down + v10_down)/(v00_down + v10_down + v01 + v11)
        
    elif metric=='TPR':
        
        TS_down = (v11 + v10_down)/(v00_up + v10_down + v01 + v11) -\
                 (v11 + v01)/(v00_down + v10_down + v01 + v11)
        
        TS_up = (v11 + v10_up)/(v00_down+v10_up+v01+v11) -\
                (v11 + v01)/(v00_up+v10_up+v01+v11)
        
    elif metric=='TNR':
       
        TS_down = (w00_down+w01)/(w00_down+w10_up+w01+w11) -\
                  (w00_up+w10_up)/(w00_up+w10_up+w01+w11)
        
        TS_up = (w00_up+w01)/(w00_up+w10_down+w01+w11) -\
                (w00_down+w10_down)/(w00_down+w10_down+w01+w11)  
    
    elif metric=='FPR':
        
        TS_down = (w10_down+w11)/(w00_up+w10_down+w01+w11) -\
                  (w01+w11)/(w00_down+w10_down+w01+w11)
        
        TS_up = (w10_up+w11)/(w00_down+w10_up+w01+w11) -\
                (w01+w11)/(w00_up+w10_up+w01+w11)
        
    elif metric=='ACCURACY':
        
        TS_down = (w00_down+v10_down + w01+v11) - (w00_up+w10_up+v11+v01)
        
        TS_up =   (w00_up + w01 + v10_up + v11) - (w00_down+w10_down+v11+v01)
        
    elif metric=='PV':
        
        TS_down = u[0,0]*(w00_down+w01)+u[0,1]*(v00_down+v01) + u[1,0]*(w10_down+w11) + u[1,1]*(v10_down+v11) -\
                  (u[0,0]*(w00_up+w10_up)+u[0,1]*(v00_up+v10_up) + u[1,0]*(w01+w11) + u[1,1]*(v01+v11))
            
        TS_up = u[0,0]*(w00_up+w01)+u[0,1]*(v00_up+v01) + u[1,0]*(w10_up+w11) + u[1,1]*(v10_up+v11) -\
                (u[0,0]*(w00_down+w10_down)+u[0,1]*(v00_down+v10_down) + u[1,0]*(w01+w11) + u[1,1]*(v01+v11))
    
    elif metric=='PV_cost':
        
        TS_down = u[0,0]*(w00_up+w01)+u[0,1]*(v00_up+v01) + u[1,0]*(w10_up+w11) + u[1,1]*(v10_up+v11) -\
                  (u[0,0]*(w00_down+w10_down)+u[0,1]*(v00_down+v10_down) + u[1,0]*(w01+w11) + u[1,1]*(v01+v11))
            
        TS_up = u[0,0]*(w00_down+w01)+u[0,1]*(v00_down+v01) + u[1,0]*(w10_down+w11) + u[1,1]*(v10_down+v11) -\
                (u[0,0]*(w00_up+w10_up)+u[0,1]*(v00_up+v10_up) + u[1,0]*(w01+w11) + u[1,1]*(v01+v11))
    
    return TS_down, TS_up

def oracle_regret(data, u, metric):
    
    A, D, Y = data['A'], data['D'], data['Y']
    
    if metric=='FNR':
        regret = ((A==0) & (Y==1)).mean() / (Y==1).mean() - ((D==0) & (Y==1)).mean() / (Y==1).mean()
        
    if metric=='TNR':
        regret = ((A==0) & (Y==0)).mean() / (Y==0).mean() - ((D==0) & (Y==0)).mean() / (Y==0).mean()

    elif metric=='TPR':
        regret = ((A==1) & (Y==1)).mean() / (Y==1).mean() - ((D==1) & (Y==1)).mean() / (Y==1).mean()
        
    elif metric=='FPR':
        regret = ((A==1) & (Y==0)).mean() / (Y==0).mean() - ((D==1) & (Y==0)).mean() / (Y==0).mean()
        
    elif metric=='ACCURACY':
        regret = (A==Y).mean() - (D==Y).mean()
        
    elif metric=='PV' or metric=='PV_cost':
        Vpi = u[0,0]*((A==0) & (Y==0)).mean() + u[0,1]*((A==0) & (Y==1)).mean() + u[1,0]*((A==1) & (Y==0)).mean() + u[1,1]*((A==1) & (Y==1)).mean()
        Vpi0 = u[0,0]*((D==0) & (Y==0)).mean() + u[0,1]*((D==0) & (Y==1)).mean() + u[1,0]*((D==1) & (Y==0)).mean() + u[1,1]*((D==1) & (Y==1)).mean()
        
        regret = Vpi-Vpi0
    
    return regret
  
def get_iv_bounds(data, dgp, a, true_class=1):
    
    X, Z, A, D, Y = data['X'], data['Z'], data['A'], data['D'], data['Y']
    nz, N = dgp['nz'], data['X'].shape[0]
    if true_class==0:
        Y = 1-Y
    
    mu_down_z = np.zeros((nz, N))
    mu_up_z = np.zeros((nz, N))

    # Compute upper and lower bounds on mu(a,x)
    for z in range(nz):
        e1_z = f_e1(X, dgp['wd'], z, dgp['beta'])
        e0_z = 1-e1_z
        
        if true_class==1:
            mu1 = f_mu(X, dgp['w_mu1'])
        else:
            mu1 = 1-f_mu(X, dgp['w_mu1'])

        mu_down_z[z] = e1_z*mu1
        mu_up_z[z] = e0_z + e1_z*mu1

    mu_down = mu_down_z.max(axis=0)
    mu_up = mu_up_z.min(axis=0)

    z_down= np.zeros((nz, N))
    z_up= np.zeros((nz, N))

    for z in range(nz):

        e1_z = f_e1(X, dgp['wd'], z, dgp['beta'])
        e0_z = 1-e1_z
        if true_class==1:
            mu1 = f_mu(X, dgp['w_mu1'])
        else:
            mu1 = 1-f_mu(X, dgp['w_mu1'])

        lower = (mu_down - mu1*e1_z)/e0_z
        upper = (mu_up - mu1*e1_z)/e0_z

        marginal = ((Z==z) & (A==a) & (D==0)).mean()*(1/N)

        z_down[z] = np.clip(lower, a_min=0, a_max=1)*marginal
        z_up[z] = np.clip(upper, a_min=0, a_max=1)*marginal

    # Compute overall bounds on unobserved cells
    v_down = z_down.sum(axis=0).sum(axis=0)
    v_up = z_up.sum(axis=0).sum(axis=0)
    
    return v_down, v_up

def compare_bounds(data, dgp, tag, pg, u, metric, id_strategies, run=0):
    
    R_star = oracle_regret(data, u, metric=metric)
    results = []
    
    for strategy in id_strategies: 
        
        TS_down, TS_up = two_step_bounds(data, dgp, u, metric=metric, id_strategy=strategy)
        OS_down, OS_up = one_step_bounds(data, dgp, u, metric=metric, id_strategy=strategy)
        
        results.append({
            'TS_down': TS_down,
            'TS_up': TS_up,
            'OS_down': OS_down,
            'OS_up': OS_up,
            'ID_type': strategy,
            'SR': data['D'].mean(),
            'pG': pg,
            'tag': tag,
            'R': R_star,
            'run': run,
            'metric': metric
        })
        
    return results

def unobs_quadrant_bounds(data, dgp, id_strategy):
    
    A, D, Y = data['A'], data['D'], data['Y']
    
    if id_strategy == 'IV':
        v10_down, v10_up = get_iv_bounds(data, dgp, a=1, true_class=1)
        v00_down, v00_up = get_iv_bounds(data, dgp, a=0, true_class=1)
        
        w10_down, w10_up = get_iv_bounds(data, dgp, a=1, true_class=0)
        w00_down, w00_up = get_iv_bounds(data, dgp, a=0, true_class=0) 
        
    elif id_strategy == 'Manski':
        v10_down, v00_down = 0, 0
        v10_up = ((A==1) & (D==0)).mean()
        v00_up = ((A==0) & (D==0)).mean()
        
        w10_down, w00_down = 0, 0
        w10_up = ((A==1) & (D==0)).mean()
        w00_up = ((A==0) & (D==0)).mean()
        
    elif id_strategy == 'MSM':
        lam = dgp['lambda']
        
        # Empirical estimates of identified terms
        v11 = ((D==1) & (A==1) & (Y==1)).mean()
        v01 = ((D==1) & (A==0) & (Y==1)).mean()
        w11 = ((D==1) & (A==1) & (Y==0)).mean()
        w01 = ((D==1) & (A==0) & (Y==0)).mean()
        
        rho10 = ((A==1) & (D==0)).mean()
        rho11 = ((A==1) & (D==1)).mean()
        
        v10_down = (1/lam)*((v11*rho10)/(rho11))
        v10_up = lam*((v11*rho10)/(rho11))
        v00_down = (1/lam)*((v01*rho10)/(rho11))
        v00_up = lam*((v01*rho10)/(rho11))
        
        w10_down = (1/lam)*((w11*rho10)/(rho11))
        w10_up = lam*((w11*rho10)/(rho11))
        w00_down = (1/lam)*((w01*rho10)/(rho11))
        w00_up = lam*((w01*rho10)/(rho11))
        
        
    return {
        'v10_down': v10_down,
        'v10_up': v10_up,
        'v00_down': v00_down,
        'v00_up': v00_up,
        'w10_down': w10_down,
        'w10_up': w10_up, 
        'w00_down': w00_down, 
        'w00_up': w00_up
    }
      