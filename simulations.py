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