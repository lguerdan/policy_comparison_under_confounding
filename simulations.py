from bound_funcs import * 


def run_sigmoid_dgp_simulation(dgp, u, metric, id_strategies, n_runs):
    
    results = []
    
    for run in range(n_runs):
        
        data = sigmoid_dgp(dgp)
        
        gx=1
        for a in [0, 1]:
            for r in [0, 1]:
                
                
                g = ((data['age']==a) & (data['race']==r))

                data_g = {
                    'X': data['X'][g],
                    'Z': data['Z'][g],
                    'D': data['D'][g],
                    'Y': data['Y'][g],
                    'A': data['A'][g],
                }
                
                outmean = ((data['D'][g] == 0) & (data['A'][g] == 0)).mean()

                tag = f'G{gx}'
                gx=gx+1
                results.extend(compare_bounds(data_g, dgp, tag, g.mean(), u, metric, id_strategies, run))

        results.extend(compare_bounds(data, dgp, 'Population', 1, u, metric, id_strategies, run))
    
    return pd.DataFrame.from_dict(results) 

def pv_simulation(dgp, id_strategies, n_runs):
    
    results = []

    for fn_cost in range(1,21):
        u = np.array([[0,-fn_cost], [-1, 0]])
        regret_runs = run_sigmoid_dgp_simulation(dgp, u, 'PV_cost', id_strategies, n_runs=n_runs)
        regret_runs['cost_ratio'] = 1/fn_cost
        if 'lambda' in dgp:
            regret_runs['lambda'] = dgp['lambda']
        results.append(regret_runs)

    for fp_cost in range(1,21):
        u = np.array([[0,-1], [-fp_cost, 0]])
        regret_runs = run_sigmoid_dgp_simulation(dgp, u, 'PV_cost', id_strategies, n_runs=n_runs)
        regret_runs['cost_ratio'] = fp_cost/1
        if 'lambda' in dgp:
            regret_runs['lambda'] = dgp['lambda']
        results.append(regret_runs)
        
    return results

def DAY_bernoulli_simulation(N, id_strategies, metrics, n_runs, lam=1):
    
    interval = np.arange(0, 1,.05)[1:]
    u = np.array([[0,0], [0, 0]])
    results = []
    
    for pD in interval:
        for pA in interval:
            for pY in interval:

                dgp = {
                    'pD': pD,
                    'pA': pA,
                    'pY': pY,
                    'N': N,
                    'lambda': lam if 'MSM' in id_strategies else 1
                }

                data, vstats = bernoulli_3d(dgp)
                ASR = ((data['A'] == 0) & (data['D'] == 0)).mean()
                ASD = (data['D'] == 0).mean()

                config = {
                    'ASR': ASR,
                    'ASD': ASD,
                    'pD': pD,
                    'pA': pA,
                    'pY': pY
                }

                for metric in metrics:
                    for run in range(n_runs):
                        bounds = compare_bounds(data, dgp, 'Population', 1, u, metric, id_strategies, run)
                        results.append({**vstats, **bounds[0], **config})
    return results