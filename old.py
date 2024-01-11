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

        RMAG =  np.random.binomial(1, .8, size=dgp['N'])
        DA_corr =  np.random.binomial(1, .6, size=dgp['N'])

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

