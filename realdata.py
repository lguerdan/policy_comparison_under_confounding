import pandas as pd
import numpy as np
import utils, estimation, vset, bounds

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def get_screening_dgp(fname, n_screen_out):

    # Load dataset
    df = pd.read_csv(fname)
    screen_in = df[df['program_enrolled_t'] == 1]
    screen_out = df[df['program_enrolled_t'] == 0].iloc[:n_screen_out]
    df = pd.concat([screen_in, screen_out]).sample(frac=1).reset_index(drop=True)
    feats = [i for i in df.columns.tolist() if 'tm1' in i]
    
    # Calculate the number of samples for the training set (40% of the total)
    train_size = int(len(df) * 0.2)

    # Split the data into training and testing sets
    train_df, df = df.iloc[:train_size], df.iloc[train_size:]
    model, scaler = LinearRegression(), StandardScaler()
    df_scaled = scaler.fit_transform(train_df[feats])
    model.fit(df_scaled, train_df['cost_t'])
    test_feats = scaler.transform(df[feats])
    
    # Set up an auditing sample on the remaining dataset
    data = {}
    feats = ['gagne_sum_tm1', 'renal_elixhauser_tm1', 'alcohol_elixhauser_tm1', 'hypertension_elixhauser_tm1',
             'dem_age_band_18-24_tm1', 'dem_age_band_25-34_tm1', 'dem_age_band_35-44_tm1', 'dem_age_band_45-54_tm1',
             'dem_age_band_55-64_tm1', 'dem_age_band_65-74_tm1', 'dem_age_band_75+_tm1']
    
    # Threshold scores at the 40th percentile
    y_hat_test = model.predict(test_feats)
    y_hat_train = model.predict(df_scaled)
    cuttoff = np.quantile(y_hat_train, .50)
    
    data['T'] = (y_hat_test > cuttoff).astype(int)
    data['Y'] = (df['gagne_sum_t'] > 0).astype(int).to_numpy()
    data['D'] = df['program_enrolled_t'].to_numpy()
    data['XU'] = df[feats].to_numpy()
    data['Z'] = np.ones_like(df['cost_t'])
    
    # Save demographic variables in the dataset
    df['65p'] = ((df['dem_age_band_65-74_tm1'] == 1) | (df['dem_age_band_75+_tm1'] == 1)).astype(int)
    
    for race in df['race'].unique():
        for age in df['65p'].unique():
            age_desc = '65+ y.o.' if age==1 else '<65 y.o.'
            data[f'x_{race}_{age_desc}'] = ((df['race'] == race) & (df['65p'] == age)).astype(int).to_numpy()
    
    return data

def get_group_descs(gdfm, sr_info):

    group_descs = []

    for g, sr, size in zip(gdfm['g'].to_list(), gdfm['selection_rate'].to_list(), gdfm['size'].to_list()):

        base = ' '.join(g.split('_')[1:]).capitalize()
        group_descs.append(base)

    return group_descs

def subgroup_exp(data, dgp, n_sims=20, K=2, n_screen_out=2000):

    bound_lists = []

    
    groups = [c for c in data.keys() if 'x_' in c]

    for s in range(n_sims):

        u = np.array([[1,0], [0, 1]])

        for k in range(K):

            in_folds, out_folds = utils.k_fold_split_and_complement(data.copy(), K)

            # Set-up datasets for fold K
            in_data, out_data = in_folds[k], out_folds[k]
            in_dgp, out_dgp = dgp.copy(), dgp.copy()
            in_dgp['N'] = in_data['XU'].shape[0]
            out_dgp['N'] = out_data['XU'].shape[0]

            # Learn models, then run inference via data from fold k
            in_probs = estimation.plugin_nuisance_probs(in_dgp, out_dgp, in_data, out_data)

            # First add the population regret
            Vpf_down, Vpf_up = vset.get_vset(in_dgp, in_data, in_probs, id_method='MSM', est_method='plugin')
            bdf = bounds.get_bounds(in_data, Vpf_down, Vpf_up, u, verbose=False)
            bdf['g'] = '_Population'
            bdf['s'] = s
            bdf['selection_rate'] = in_data['D'].mean()
            bdf['size'] = 1
            bound_lists.append(bdf)

            # Add subgroup-level regret
            for g in groups: 
                gdata = utils.mask_dictionary_arrays(in_data.copy(), in_data[g])
                gprobs = utils.mask_dictionary_arrays(in_probs.copy(), in_data[g])

                # After learning the nuisance functions, we need to evaluate this over each subgroup
                Vpf_down, Vpf_up = vset.get_vset(in_dgp, gdata, gprobs, id_method='MSM', est_method='plugin')
                bdf = bounds.get_bounds(gdata, Vpf_down, Vpf_up, u, verbose=False)
                bdf['g'] = g
                bdf['s'] = s
                bdf['selection_rate'] = (gdata['D'] == 1).mean()
                bdf['size'] = gdata['XU'].shape[0] / in_data['XU'].shape[0]
                bound_lists.append(bdf)

    return pd.concat(bound_lists).groupby(['g', 's', 'metric']).mean().reset_index()
