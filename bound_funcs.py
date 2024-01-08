from dgp import *
import pandas as pd
import numpy as np


# Tolerance parameter to prevent division by zero in computation of regret metrics
eps = .001

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



########### DATA STRUCTURE NOTES ############
# v[y,t,d]
# - contains oracle values for v[y,t,0] if they exist
# - contains values for v[y,t,1]

# Vpf_down = [y,t]
# - contains upper bound for v_y(t,0) in each entry

# Vpf_down = [y,t]
# - contains lower bound for v_y(t,0) in each entry


def delta_bounds(v, Vpf_down, Vpf_up, u, metric):

    # given: v, Vpf_down, Vpf_up

    v_up = v.copy()
    v_down = v.copy()
    v = v.copy()

    if metric=='m_u':

        yu = ((u[1,1] - u[0,1]) > (u[1,0] - u[0,0])).astype(int)
        v_up[yu, 1, 0] = Vpf_up[yu,1]
        v_up[1-yu, 1, 0] = Vpf_down[1-yu,1]

        v_down[yu, 1, 0] = Vpf_down[yu,1]
        v_down[1-yu, 1, 0] = Vpf_up[1-yu,1]

        # Apply delta regret definition
        R_down = sum((u[a, y] - u[1-a, y]) * v_down[y, a, 1-a] for a in [0, 1] for y in [0, 1])
        R_up = sum((u[a, y] - u[1-a, y]) * v_up[y, a, 1-a] for a in [0, 1] for y in [0, 1])

    if metric=='m_y=1' or metric=='m_y=0':

        #y=1 imples TPR, y=0 implies FPR
        y = 1 if metric=='m_y=1' else 0

        # Apply arg min/max, max to upper bound
        v_up[y, 1, 0] = Vpf_up[y,1]
        v_up[y, 0, 0] = Vpf_down[y, 0] if v_up[y, 1, 0] > v_up[y, 0, 1] else Vpf_up[y, 0]

        # Apply arg min/max, max to lower bound
        v_down[y, 1, 0] = Vpf_down[y,1]
        v_down[y,0,0] = Vpf_up[y, 0] if v_down[y, 1, 0] > v_down[y, 0, 1]  else Vpf_down[y, 0]

        # Apply delta regret definition
        R_up = (v_up[y,1,0] - v_up[y,0,1])/(v_up[y,0,0] + v_up[y,1,0] + v_up[y,0,1] + v_up[y,1,1])
        R_down = (v_down[y,1,0] - v_down[y,0,1])/(v_down[y,0,0] + v_down[y,1,0] + v_down[y,0,1] + v_down[y,1,1])

    if metric=='m_a=1':
        a = 1

        # May need to populate v[y,t,0] terms from dgp later
        pD = np.clip(v[0,0,a] + v[1,0,a] + v[0,1,a] + v[1,1,a], eps, 1-eps)
        pT = np.clip(v[0,a,0] + v[1,a,0] + v[0,a,1] + v[1,a,1], eps, 1-eps)
        rho_10 = v[0,1,0] + v[1,1,0]
        rho_01 = v[0,0,1] + v[1,0,1]

        sigma_a = (1-2*a) * (rho_10 - rho_01)

        # Apply arg min/max, max to upper bound
        v_up[1,1,0] = Vpf_up[1,1]
        v_down[1,1,0] = Vpf_down[1,1]

        # Apply delta regret definition
        R_up = (sigma_a * v_up[a,a,a] + pD * v_up[a, a, 1-a] - pT * v_up[a,1-a,a]) / (pD * pT)
        R_down = (sigma_a * v_down[a,a,a] + pD * v_down[a, a, 1-a] - pT * v_down[a,1-a,a]) / (pD * pT)


    if metric=='m_a=0':
        a = 0

        # May need to populate v[y,t,0] terms from dgp later
        pD = np.clip(v[0,0,a] + v[1,0,a] + v[0,1,a] + v[1,1,a], eps, 1-eps)
        pT = np.clip(v[0,a,0] + v[1,a,0] + v[0,a,1] + v[1,a,1], eps, 1-eps)
        
        rho_10 = v[0,1,0] + v[1,1,0]
        rho_01 = v[0,0,1] + v[1,0,1]

        sigma_a = (1-2*a) * (rho_10 - rho_01)

        # Apply arg min/max, max to upper bound
        v_up[0,1,0] = Vpf_down[0,1]
        v_down[0,1,0] = Vpf_up[0,1]
        v_up[0,0,0] = Vpf_up[0,0] if sigma_a >= 0 else Vpf_down[0,0]
        v_down[0,0,0] = Vpf_down[0,0] if sigma_a >= 0 else Vpf_up[0,0]

        # Apply delta regret definition
        R_up = (sigma_a * v_up[a,a,a] + pD * v_up[a, a, 1-a] - pT * v_up[a,1-a,a]) / (pD * pT)
        R_down = (sigma_a * v_down[a,a,a] + pD * v_down[a, a, 1-a] - pT * v_down[a,1-a,a]) / (pD * pT)
 
    return R_down, R_up


def standard_bounds(v, Vpf_down, Vpf_up, u, metric):

    #prevent division by zero
    Vpf_down = np.clip(Vpf_down, eps, 1-eps)
    v = v.copy()
    v_up = v.copy()
    v_down = v.copy()

    if metric=='m_u':

        y_ta = (u[0,0] < u[0,1]).astype(int)
        y_tb = (u[1,0] < u[1,1]).astype(int)

        # max, max m_u(v0,v1;pi)
        vu = v.copy()
        vu[y_ta,0,0] = Vpf_up[y_ta,0]
        vu[1-y_ta,0,0] = Vpf_down[1-y_ta,0]

        vu[y_tb,1,0] = Vpf_up[y_tb,1]
        vu[1-y_tb,1,0] = Vpf_down[1-y_tb,1]

        m_u_pi_up = sum(u[t, y] * (vu[y,t,0] + vu[y,t,1]) for t in [0, 1] for y in [0, 1])

        # min, min m_u(v0,v1;pi)
        vu = v.copy()
        vu[y_ta,0,0] = Vpf_down[y_ta,0]
        vu[1-y_ta,0,0] = Vpf_up[1-y_ta,0]

        vu[y_tb,1,0] = Vpf_down[y_tb,1]
        vu[1-y_tb,1,0] = Vpf_up[1-y_tb,1]

        m_u_pi_down = sum(u[t, y] * (vu[y,t,0] + vu[y,t,1]) for t in [0, 1] for y in [0, 1])

        # max, max m_u(v0,v1;pi_0)
        vu = v.copy()
        vu[y_ta,0,0] = Vpf_up[y_ta,0]
        vu[1-y_ta,0,0] = Vpf_down[1-y_ta,0]

        vu[y_ta,1,0] = Vpf_up[y_ta,1]
        vu[1-y_ta,1,0] = Vpf_down[1-y_ta,1]

        m_u_pi0_up = sum(u[d, y] * (vu[y,0,d] + vu[y,1,d]) for d in [0, 1] for y in [0, 1])

        # min, min m_u(v0,v1;pi_0)
        vu[y_ta,0,0] = Vpf_down[y_ta,0]
        vu[1-y_ta,0,0] = Vpf_up[1-y_ta,0]

        vu[y_ta,1,0] = Vpf_down[y_ta,1]
        vu[1-y_ta,1,0] = Vpf_up[1-y_ta,1]

        m_u_pi0_down = sum(u[d, y] * (vu[y,0,d] + vu[y,1,d]) for d in [0, 1] for y in [0, 1])

        R_up = m_u_pi_up - m_u_pi0_down
        R_down = m_u_pi_down - m_u_pi0_up

    if metric=='m_y=1' or metric=='m_y=0':

        y = 1 if metric=='m_y=1' else 0
        assert v[y,:,:].sum() > 0, 'Zero base rate denominator'

        R_up = (Vpf_up[y,1] + v[y,1,1])/ (Vpf_down[y,0] + v[y,0,1] + Vpf_up[y,1] + v[y,1,1]) \
            - (v[y,0,1] + v[y,1,1]) / (Vpf_up[y,0]  + v[y,0,1] + Vpf_up[y,1] + v[y,1,1])

        R_down = (Vpf_down[y,1] + v[y,1,1])/ (Vpf_up[y,0] + v[y,0,1] + Vpf_down[y,1] + v[y,1,1]) \
            - (v[y,0,1] + v[y,1,1]) / (Vpf_down[y,0]  + v[y,0,1] + Vpf_down[y,1] + v[y,1,1])

    if metric=='m_a=1':
        a=1
        pD = np.clip(v[0,0,a] + v[1,0,a] + v[0,1,a] + v[1,1,a], eps, 1-eps)
        pT = np.clip(v[0,a,0] + v[1,a,0] + v[0,a,1] + v[1,a,1], eps, 1-eps)

        assert pD > 0, 'PD=0'
        assert pT > 0, 'pT=0'

        R_up = (Vpf_up[1,1] + v[1,1,1])/pT - (v[1,0,1] + v[1,1,1])/pD
        R_down = (Vpf_down[1,1] + v[1,1,1])/pT - (v[1,0,1] + v[1,1,1])/pD

    if metric=='m_a=0':
        a=0
        pD = np.clip(v[0,0,a] + v[1,0,a] + v[0,1,a] + v[1,1,a], eps, 1-eps)
        pT = np.clip(v[0,a,0] + v[1,a,0] + v[0,a,1] + v[1,a,1], eps, 1-eps)

        R_up = (Vpf_up[0,0] + v[0,0,1])/pT - (Vpf_down[0,0] + Vpf_down[0,1])/pD
        R_down = (Vpf_down[0,0] + v[0,0,1])/pT - (Vpf_up[0,0] + Vpf_up[0,1])/pD

    return R_down, R_up


def oracle_regret(v, u, metric):
    # This function assumes knowledge of[y,t,0]

    v = v.copy()

    if metric=='m_y=1' or metric=='m_y=0':
        y = 1 if metric=='m_y=1' else 0
        regret = (v[y,1,0] - v[y,0,1]) / (v[y,0,0] + v[y,1,0] + v[y,0,1] + v[y,1,1])

    if metric=='m_u':
        regret = sum([(u[a,y] - u[1-a,y]) * v[y,a,1-a] for a in [0, 1] for y in [0, 1]])

    if metric=='m_a=1' or metric=='m_a=0':
        a = 1 if metric=='m_a=1' else 0
        pD = np.clip(v[0,0,a] + v[1,0,a] + v[0,1,a] + v[1,1,a], eps, 1-eps)
        pT = np.clip(v[0,a,0] + v[1,a,0] + v[0,a,1] + v[1,a,1], eps, 1-eps)

        if a == 1:
            regret = (v[1,1,0] + v[1,1,1])/pT -  (v[1,0,1] + v[1,1,1])/pD

        if a == 0:
            regret = (v[0,0,0] + v[0,0,1])/pT -  (v[0,0,0] + v[0,1,0])/pD

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
        
        TS_down, TS_up, ubs = two_step_bounds(data, dgp, u, metric=metric, id_strategy=strategy)
        OS_down, OS_up, ubs = one_step_bounds(data, dgp, u, metric=metric, id_strategy=strategy)
        
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
            'metric': metric,
            **ubs
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

        v10_down, v10_up = get_msm_bounds(data, dgp, a=1, true_class=1)
        v00_down, v00_up = get_msm_bounds(data, dgp, a=0, true_class=1)

        v10_down, v10_up = f_e1(X, dgp['wd'], 4, dgp['beta'])
        v00_down, v00_up = f_e1(X, dgp['wd'], 4, dgp['beta'])

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

    def check_bounds(data, Vpf_down, Vpf_up):
        '''Given observational data (Y,D,T) ~ p() compute bounds across measures of interest.'''
        
        Y, D, T = data['Y'], data['D'], data['T']
        v = np.zeros((2,2,2))

        for y in range(2):
            for d in range(2):
                for t in range(2):
                    v[y,t,d] = ((Y==y) & (D==d) & (T==t)).mean()
        
        metrics = ['m_y=1', 'm_y=0', 'm_a=0', 'm_a=1', 'm_u']
        u = np.array([[1,0], [0, 1]])

        for metric in metrics:
            
            R_oracle = oracle_regret(v, u, metric)
            Rs_down, Rs_up = standard_bounds(v, Vpf_down, Vpf_up, u, metric)
            Rd_down, Rd_up = delta_bounds(v, Vpf_down, Vpf_up, u, metric)

            print(f'metric: {metric}')
            print(f'Standard bounds [{Rs_down:.3}, {Rs_up:.3}]')
            print(f'Delta bounds: [{Rd_down:.3}, {Rd_up:.3}]')
            print(f'Oracle: {R_oracle:.4}')
            print()