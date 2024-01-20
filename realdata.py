import pandas as pd
import numpy as np

def get_screening_dgp(fname, n_screen_out):

    df = pd.read_csv(fname)
    screen_in = df[df['program_enrolled_t'] == 1]
    screen_out = df[df['program_enrolled_t'] == 0].iloc[:n_screen_out]
    df = pd.concat([screen_in, screen_out])

    feats = [i for i in df.columns.tolist() if 'tm1' in i]
    data = {}
    data['Y'] = (df['gagne_sum_t'] > 0).astype(int).to_numpy()
    data['D'] = df['program_enrolled_t'].to_numpy()
    data['XU'] = df[feats].to_numpy()
    data['Z'] = np.ones_like(df['cost_t'])
    
    # TODO: construct different T's
    data['T'] = (df['cost_t'] > 3000).to_numpy()
    
    df['65p'] = ((df['dem_age_band_65-74_tm1'] == 1) | (df['dem_age_band_75+_tm1'] == 1)).astype(int)
    
    for race in df['race'].unique():
        for age in df['65p'].unique():
            age_desc = '65+' if age==1 else '<65'
            data[f'x_{race}_{age_desc}'] = ((df['race'] == race) & (df['65p'] == age)).astype(int).to_numpy()

    
    return data

def get_group_descs(gdfm):

    group_descs = []

    for g, sr, size in zip(gdfm['g'].to_list(), gdfm['selection_rate'].to_list(), gdfm['size'].to_list()):

        base = ' '.join(g.split('_')[1:]).capitalize()
        base += f', $\psi$={sr:.2}, $\gamma$={size:.2} '

        group_descs.append(base)

    return group_descs