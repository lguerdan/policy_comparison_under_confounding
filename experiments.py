import numpy as np
import pandas as pd
import statsmodels.stats.api as sms

import utils, estimation
import dgp as datagen


def estimation_coverage_experiment(msm_dgp, Ns, Nsims=30):

    superset = datagen.generate_data(msm_dgp)
    coverage_results = []

    for n in Ns:
        msm_dgp['N'] = n
        for sim in range(Nsims):
            data = utils.sample_arrays_common_indices(superset.copy(), n)

            dr_results = estimation.estimate_bounds(msm_dgp, data, id_method='MSM', est_method='dr', K=5)
            dr_results['N'] = n
            dr_results['S'] = sim
            dr_results['method'] = 'dr'
            coverage_results.append(dr_results)

            plugin_results = estimation.estimate_bounds(msm_dgp, data, id_method='MSM', est_method='plugin', K=5)
            plugin_results['N'] = n
            plugin_results['S'] = sim
            plugin_results['method'] = 'plugin'
            coverage_results.append(plugin_results)

            oracle_results = estimation.estimate_bounds(msm_dgp, data, id_method='MSM', est_method='oracle')
            oracle_results['N'] = n
            oracle_results['S'] = sim
            oracle_results['method'] = 'oracle'
            coverage_results.append(oracle_results)
            
    return pd.concat(coverage_results)


def get_est_exp_metadata(coveragedf, Ns):

    metrics = ['m_y=1', 'm_y=0', 'm_a=0', 'm_a=1', 'm_u']
    N_results = {
        'R_oracle': [],
        'Rd_up_oracle': [],
        'Rd_up_pl_mean': [],
        'Rd_up_pl_ci': [],
        'Rd_down_oracle': [],
        'Rd_down_pl_mean': [],
        'Rd_down_pl_ci': [],
        'Rs_up_oracle': [],
        'Rs_up_pl_mean': [],
        'Rs_up_pl_ci': [],
        'Rs_down_oracle': [],
        'Rs_down_pl_mean': [],
        'Rs_down_pl_ci': [],
        'N': [],
        'metric': [],
    }

    for n in Ns:
        for metric in metrics:

            oracle = coveragedf[(coveragedf['metric'] == metric) & (coveragedf['N'] == n) & (coveragedf['method'] == 'oracle')]
            plugin = coveragedf[(coveragedf['metric'] == metric) & (coveragedf['N'] == n) & (coveragedf['method'] == 'plugin')]

            N_results['R_oracle'].append(oracle['R_oracle'].mean())
            N_results['metric'].append(metric)
            N_results['N'].append(n)

            plvals_up = plugin['Rs_up'].tolist()
            pl_ci_up = sms.DescrStatsW(plvals_up).tconfint_mean()
            pl_plvals_down = plugin['Rs_down'].tolist()
            pl_ci_down = sms.DescrStatsW(pl_plvals_down).tconfint_mean()

            N_results['Rs_up_oracle'].append(oracle['Rs_up'].mean())
            N_results['Rs_up_pl_mean'].append(plugin['Rs_up'].mean())
            N_results['Rs_up_pl_ci'].append(pl_ci_up[1]-pl_ci_up[0])

            N_results['Rs_down_oracle'].append(oracle['Rs_down'].mean())
            N_results['Rs_down_pl_mean'].append(plugin['Rs_down'].mean())
            N_results['Rs_down_pl_ci'].append(pl_ci_down[1]-pl_ci_down[0])

            plvals_up = plugin['Rd_up'].tolist()
            pl_ci_up = sms.DescrStatsW(plvals_up).tconfint_mean()
            pl_plvals_down = plugin['Rd_down'].tolist()
            pl_ci_down = sms.DescrStatsW(pl_plvals_down).tconfint_mean()

            N_results['Rd_up_oracle'].append(oracle['Rd_up'].mean())
            N_results['Rd_up_pl_mean'].append(plugin['Rd_up'].mean())
            N_results['Rd_up_pl_ci'].append(pl_ci_up[1]-pl_ci_up[0])

            N_results['Rd_down_oracle'].append(oracle['Rd_down'].mean())
            N_results['Rd_down_pl_mean'].append(plugin['Rd_down'].mean())
            N_results['Rd_down_pl_ci'].append(pl_ci_down[1]-pl_ci_down[0])
            
    return pd.DataFrame(N_results)


def relevance_sensitivity_experiment(dgp, beta_zd, n_sims, est_method='plugin', K=5):

    relevance_bounds = []
    for zd in beta_zd:
        dgp['beta_zd'] = zd
        
        for sim in range(n_sims):
            data = datagen.generate_data(dgp)
            bounds = estimation.estimate_bounds(dgp, data, id_method='IV', est_method=est_method, K=K)
            bounds['beta_zy'] = zd
            relevance_bounds.append(bounds)

    return pd.concat(relevance_bounds)


def exclusion_sensitivity_experiment(dgp, beta_zy, n_sims, est_method='plugin', K=5):

    exclusion_bounds = []
    for zy in beta_zy:
        dgp['beta_zy'] = zy
        
        for sim in range(n_sims):
            data = datagen.generate_data(dgp)    
            bounds = estimation.estimate_bounds(dgp, data, id_method='IV', est_method=est_method, K=K)
            bounds['beta_zy'] = zy
            exclusion_bounds.append(bounds)

    return pd.concat(exclusion_bounds)

def design_sensitivity_exp(dgp, data, lambdas, n_sims=10):

    lambda_bounds = []

    for lam in lambdas:
        for sim in range(n_sims):

            dgp['lambda'] = lam
            bounds = estimation.estimate_bounds(dgp, data, id_method='MSM', est_method='plugin', K=5)
            bounds['lambda'] = lam
            lambda_bounds.append(bounds)

    return pd.concat(lambda_bounds)


def cost_ratio_sweep_exp(dgp, data, lam, n_sims=5):
    
    dgp['lambda'] = lam
    regret_runs = []

    for n in range(n_sims):

      for fn_cost in range(1,25):
          u = np.array([[0,-fn_cost], [-1, 0]])
          result = estimation.sample_split_crossfit(dgp, data, id_method='MSM', est_method='plugin', K=5, u=u)
          result['cr'] = 1/fn_cost
          result['lambda'] = dgp['lambda']
          regret_runs.append(result)

      for fp_cost in range(1,25):
          u = np.array([[0,-1], [-fp_cost, 0]])
          result = estimation.sample_split_crossfit(dgp, data, id_method='MSM', est_method='plugin', K=5, u=u)
          result['cr'] = fp_cost
          result['lambda'] = dgp['lambda']
          regret_runs.append(result)
          
    return pd.concat(regret_runs)

def msm_sensitivity_experiment(msm_dgp, lambda_star, n_sims=10):

    msm_bounds = []

    for ls in lambda_star:
        msm_dgp['lambda_star'] = ls
        for sim in range(n_sims):
            data = datagen.generate_data(msm_dgp)
            bounds = estimation.estimate_bounds(msm_dgp, data, id_method='MSM', est_method='plugin', K=5)
            bounds['ls'] = ls
            msm_bounds.append(bounds)

    return pd.concat(msm_bounds)
