import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import matplotlib.pyplot as plt

from bounds import * 
from plots import *


########### DATA STRUCTURE NOTES ############
# v[y,t,d]
# - contains oracle values for v[y,t,0] if they exist
# - contains values for v[y,t,1]

# Vpf_up = [y,t]
# - contains upper bound for v_y(t,0) in each entry

# Vpf_down = [y,t]
# - contains lower bound for v_y(t,0) in each entry
##############################################


def generate_pc_quadrants():
    nrange = np.arange(0.03,.97,.1)
    N = len(nrange) ** 4

    v = np.zeros((2,2,2,N))
    Vpf_down = np.ones((2,2,N))*.0001 #prevent division by zero
    Vpf_up = np.zeros((2,2,N))

    n = 0

    # Vary identified terms v[y,t,1]
    for a in nrange:
        for b in nrange:
            for c in nrange:
                for d in nrange:
                    n += 1 
                    if a + b + c + d <= 1:
                        v[0,0,1,n] = a
                        v[0,1,1,n] = b
                        v[1,0,1,n] = c
                        v[1,1,1,n] = d

    # Populate unidentified terms v[y,t,0] satisfying sum v[y,t,d] = 1. 
    # Not uniform sampling but that's fine for our purposes
    for n in range(N): 
        s = 1-v[:,:,1,n].sum()
        ubs = np.random.rand(2,2)
        v[:,:,0,n] = (s * ubs) / ubs.sum()
        
    # Populate worst case bounds Vpf_down, Vpf_up based on a constraint from p(.)   
    for n in range(N): 
        rho_10 = v[0,1,0,n] + v[1,1,0,n]
        Vpf_up[0,1,n] = rho_10 
        Vpf_up[1,1,n] = rho_10 

        rho_00 = v[0,0,0,n] + v[1,0,0,n]
        Vpf_up[0,0,n] = rho_00
        Vpf_up[1,0,n] = rho_00
          
    # Check all cells sum to one
    for n in range(N):
        assert np.abs(v[:,:,:,n].sum() - 1) < .001

    return v, Vpf_down, Vpf_up

def run_predictive_performance_coverage_tests(v, Vpf_down, Vpf_up):

    N = v.shape[3]

    u = np.array([[1,0], [0, 1]])
    metrics = ['m_y=1', 'm_y=0', 'm_a=0', 'm_a=1', 'm_u']

    gamma_0 = Vpf_up[0,0,:] + Vpf_up[0,1,:] + v[0,1,1,:] + v[0,0,1,:]
    gamma_1 = Vpf_up[1,0,:] + Vpf_up[1,1,:] + v[1,0,1,:] + v[1,1,1,:]
    pD = 1 - (v[0,0,1,:] + v[0,1,1,:] + v[1,0,1,:] + v[1,1,1,:])
    pT = 1 - (v[0,1,0,:] + v[0,1,1,:] + v[1,1,0,:] + v[1,1,1,:])
    alpha = Vpf_up[0,0,:] - Vpf_down[0,0,:]

    # Measures for regret seperation plot
    D_y1 = alpha*v[1,1,1]/ (gamma_1**2)
    D_y0 = alpha*v[0,1,1]/ (gamma_0**2)
    D_a0 = alpha/np.max([pD, pT],axis=0)
    D_u =  alpha * (u[0,0] + u[0,1])

    bounds = {
        'Rd_down': [],
        'Rd_up': [],
        'Rs_down': [],
        'Rs_up': [],
        'R_oracle': [],
        'metric': [],
        'u_00': [],
        'u_01': [],
        'u_10': [],
        'u_11': [],
        'D_y0':[],
        'D_y1':[],
        'D_a0':[],
        'D_u':[]
    }

    # Check positive and negative class performance, PPV
    for n in range(N):
        for metric in metrics: 
            Rd_down, Rd_up = delta_bounds(v[:,:,:,n], Vpf_down[:,:,n], Vpf_up[:,:,n], u, metric)
            Rs_down, Rs_up = standard_bounds(v[:,:,:,n], Vpf_down[:,:,n], Vpf_up[:,:,n], u, metric)
            R_oracle = oracle_regret(v[:,:,:,n], u, metric)
            
            try:
                assert Rd_down < Rd_up, f'Bound ordinality check: {Rd_down:.3} < {Rd_up:.3} violated for n={n}'
                assert Rs_down < Rs_up
                assert Rd_down-.02 <= R_oracle and R_oracle <= Rd_up+.02
                assert Rs_down-.02 <= R_oracle and R_oracle <= Rs_up+.02

            except: 
                print(f'metric: {metric}')
                print(f'Standard bounds [{Rs_down:.3}, {Rs_up:.3}]')
                print(f'Delta bounds: [{Rd_down:.3}, {Rd_up:.3}]')
                print(f'Oracle: {R_oracle:.4}')
                print()
                
            bounds['Rd_down'].append(Rd_down)
            bounds['Rd_up'].append(Rd_up)
            bounds['Rs_down'].append(Rs_down)
            bounds['Rs_up'].append(Rs_up)
            bounds['R_oracle'].append(R_oracle)
            bounds['metric'].append(metric)
            bounds['u_00'].append(u[0,0])
            bounds['u_01'].append(u[0,1])
            bounds['u_10'].append(u[1,0])
            bounds['u_11'].append(u[1,1])
            bounds['D_y0'].append(D_y0[n])
            bounds['D_y1'].append(D_y1[n])
            bounds['D_a0'].append(D_a0[n])
            bounds['D_u'].append(D_u[n])
        
    # Check utilities over varying ranges
    utils = [.01,.5,.9]
    for n in range(N):
        for ua in utils:
            for ub in utils:
                for uc in utils:
                    for ud in utils:
                        u = np.array([[ua,ub],[uc,ud]])
                        Rd_down, Rd_up = delta_bounds(v[:,:,:,n], Vpf_down[:,:,n], Vpf_up[:,:,n], u, 'm_u')
                        Rs_down, Rs_up = standard_bounds(v[:,:,:,n], Vpf_down[:,:,n], Vpf_up[:,:,n], u, 'm_u')
                        R_oracle = oracle_regret(v[:,:,:,n], u, 'm_u')
                        
                        try:
                            assert Rd_down <= Rd_up+.001, f'Bound ordinality check: {Rd_down:.3} < {Rd_up:.3} violated for n={n}'
                            assert Rs_down <= Rs_up+.001
                            assert Rd_down-.02 <= R_oracle and R_oracle <= Rd_up+.02
                            assert Rs_down-.02 <= R_oracle and R_oracle <= Rs_up+.02

                        except: 
                            print('N:', n)
                            print('U:', ua, ub, uc, ud)
                            print(f'Standard bounds [{Rs_down:.3}, {Rs_up:.3}]')
                            print(f'Delta bounds: [{Rd_down:.3}, {Rd_up:.3}]')
                            print(f'Oracle: {R_oracle:.4}')
                            print()
                            
                        bounds['Rd_down'].append(Rd_down)
                        bounds['Rd_up'].append(Rd_up)
                        bounds['Rs_down'].append(Rs_down)
                        bounds['Rs_up'].append(Rs_up)
                        bounds['R_oracle'].append(R_oracle)
                        bounds['metric'].append(metric)
                        bounds['u_00'].append(u[0,0])
                        bounds['u_01'].append(u[0,1])
                        bounds['u_10'].append(u[1,0])
                        bounds['u_11'].append(u[1,1])
                        bounds['D_y1'].append(D_y1[n])
                        bounds['D_y0'].append(D_y0[n])
                        bounds['D_a0'].append(D_a0[n])
                        bounds['D_u'].append(D_u[n])

    return pd.DataFrame(bounds), D_u


def run_utility_coverage_tests(v, Vpf_down, Vpf_up):

    gamma_0 = Vpf_up[0,0] + Vpf_up[0,1] + v[0,1,1] + v[0,0,1]
    gamma_1 = Vpf_up[1,0] + Vpf_up[1,1] + v[1,0,1] + v[1,1,1]
    pD = 1 - (v[0,0,1] + v[0,1,1] + v[1,0,1] + v[1,1,1])
    pT = 1 - (v[0,1,0] + v[0,1,1] + v[1,1,0] + v[1,1,1])
    alpha = Vpf_up[0,0] - Vpf_down[0,0]
    u = np.array([[1,0], [0, 1]])

    # Measures for regret seperation plot
    D_y1 = alpha*v[1,1,1]/ (gamma_1**2)
    D_y0 = alpha*v[0,1,1]/ (gamma_0**2)
    D_a0 = alpha/np.max([pD, pT],axis=0)
    D_u =  alpha * (u[0,0] + u[0,1])
    
    bounds = {
        'Rd_down': [],
        'Rd_up': [],
        'Rs_down': [],
        'Rs_up': [],
        'R_oracle': [],
        'metric': [],
        'u_00': [],
        'u_01': [],
        'u_10': [],
        'u_11': [],
        'D_y0':[],
        'D_y1':[],
        'D_a0':[],
        'D_u':[]
    }

    for u00 in range(20):
        for u11 in range(20):

            u = np.array([[u00,0],[0,u11]])

            Rd_down, Rd_up = delta_bounds(v, Vpf_down, Vpf_up, u, 'm_u')
            Rs_down, Rs_up = standard_bounds(v, Vpf_down, Vpf_up, u, 'm_u')
            R_oracle = oracle_regret(v, u, 'm_u')

            try:
                assert Rd_down <= Rd_up+.001, f'Bound ordinality check: {Rd_down:.3} < {Rd_up:.3} violated for n={n}'
                assert Rs_down <= Rs_up+.001
                assert Rd_down-.02 <= R_oracle and R_oracle <= Rd_up+.02
                assert Rs_down-.02 <= R_oracle and R_oracle <= Rs_up+.02

            except: 
                print('N:', n)
                print('U:', ua, ub, uc, ud)
                print(f'Standard bounds [{Rs_down:.3}, {Rs_up:.3}]')
                print(f'Delta bounds: [{Rd_down:.3}, {Rd_up:.3}]')
                print(f'Oracle: {R_oracle:.4}')
                print()

            bounds['Rd_down'].append(Rd_down)
            bounds['Rd_up'].append(Rd_up)
            bounds['Rs_down'].append(Rs_down)
            bounds['Rs_up'].append(Rs_up)
            bounds['R_oracle'].append(R_oracle)
            bounds['metric'].append('m_u')
            bounds['u_00'].append(u[0,0])
            bounds['u_01'].append(u[0,1])
            bounds['u_10'].append(u[1,0])
            bounds['u_11'].append(u[1,1])
            bounds['D_y1'].append(D_y1)
            bounds['D_y0'].append(D_y0)
            bounds['D_a0'].append(D_a0)
            bounds['D_u'].append((u[0,0]+u[0,1])*alpha)   

    return pd.DataFrame(bounds)


if  __name__ == "__main__":
    
    v, Vpf_down, Vpf_up = generate_pc_quadrants()
    pp_bounds, D_u = run_predictive_performance_coverage_tests(v, Vpf_down, Vpf_up)

    n=D_u.argsort()[9000]
    util_bounds = run_utility_coverage_tests(v[:,:,:,n].copy(), Vpf_down[:,:,n].copy(), Vpf_up[:,:,n].copy())

    plot_regret_seperation(pp_bounds)
    plot_util_regret_seperation(util_bounds)
