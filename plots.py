import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms
import utils, realdata

def bound_plot(metric, regret_df, regret_runs, save=False):

    sns.set(font_scale=1.3)
    plt.figure(figsize=(10, 6))
    sns.set_style("white")

    plt.axvline(0, color='grey', zorder=1, linestyle='--')

    for ix, row in regret_df.iterrows():

        # One-step IV bounds
        plt.plot([row['OS_down_iv'], row['OS_up_iv']], [ix+.25, ix+.25], color='r', label="One step (IV)", linewidth=2)
        plt.scatter(row['OS_down_iv'], ix+.25,  color='r', s=30)
        plt.scatter(row['OS_up_iv'], ix+.25, color='r', label="R")

        sd = regret_runs[regret_runs['tag'] == row['tag']]['OS_down_iv'].tolist()
        ci_d = sms.DescrStatsW(sd).tconfint_mean()
        l_d = abs(ci_d[1]-ci_d[0])
        plt.errorbar(row['OS_down_iv'], ix+.25, xerr=l_d, color='r', fmt='none', capsize=5, linewidth=2, elinewidth=2)

        su = regret_runs[regret_runs['tag'] == row['tag']]['OS_down_iv'].tolist()
        ci_u = sms.DescrStatsW(su).tconfint_mean()
        l_u = abs(ci_u[1]-ci_u[0])   
        plt.errorbar(row['OS_up_iv'], ix+.25, xerr=l_u, color='r', fmt='none', capsize=5, linewidth=2, elinewidth=2)

        # One-step WC bounds
        plt.plot([row['OS_down_wc'], row['OS_up_wc']], [ix+.12, ix+.12], color='r', linestyle='--', label="One step (WC)",  linewidth=2)
        plt.scatter(row['OS_down_wc'], ix+.12,  color='r', s=30)
        plt.scatter(row['OS_up_wc'], ix+.12, color='r', label="R")

        sd = regret_runs[regret_runs['tag'] == row['tag']]['OS_down_wc'].tolist()
        ci_d = sms.DescrStatsW(sd).tconfint_mean()
        l_d = abs(ci_d[1]-ci_d[0])
        plt.errorbar(row['OS_down_wc'], ix+.12, xerr=l_d, color='r', fmt='none', capsize=5, linewidth=2, elinewidth=2)

        su = regret_runs[regret_runs['tag'] == row['tag']]['OS_up_wc'].tolist()
        ci_u = sms.DescrStatsW(su).tconfint_mean()
        l_u = abs(ci_u[1]-ci_u[0])   
        plt.errorbar(row['OS_up_wc'], ix+.12, xerr=l_u, color='r', fmt='none', capsize=5, linewidth=2, elinewidth=2)


        plt.scatter(row['TS_down_iv'], ix-.07, color='b', s=30)
        plt.scatter(row['TS_up_iv'],ix-.07, color='b', s=30)
        plt.plot([row['TS_down_iv'], row['TS_up_iv']], [ix+-.07, ix-.07], color='b', label="Two step (IV)",  linewidth=2)

        sd = regret_runs[regret_runs['tag'] == row['tag']]['TS_down_iv'].tolist()
        ci_d = sms.DescrStatsW(sd).tconfint_mean()
        l_d = abs(ci_d[1]-ci_d[0])
        plt.errorbar(row['TS_down_iv'], ix-.07, xerr=l_d, color='b', fmt='none', capsize=5, linewidth=2, elinewidth=2)

        su = regret_runs[regret_runs['tag'] == row['tag']]['TS_up_iv'].tolist()
        ci_u = sms.DescrStatsW(su).tconfint_mean()
        l_u = abs(ci_u[1]-ci_u[0])   
        plt.errorbar(row['TS_up_iv'], ix-.07, xerr=l_u, color='b', fmt='none', capsize=5, linewidth=2, elinewidth=2)


        plt.scatter(row['TS_down_wc'], ix-.2, color='b', s=30)
        plt.scatter(row['TS_up_wc'], ix-.2, color='b', s=30)
        plt.plot([row['TS_down_wc'], row['TS_up_wc']], [ix-.2, ix-.2], color='b', linestyle='--', label="Two step (WC)",  linewidth=2)


        sd = regret_runs[regret_runs['tag'] == row['tag']]['TS_down_wc'].tolist()
        ci_d = sms.DescrStatsW(sd).tconfint_mean()
        l_d = abs(ci_d[1]-ci_d[0]) 
        plt.errorbar(row['TS_down_wc'], ix-.2, xerr=l_d, color='b', fmt='none', capsize=5, linewidth=2, elinewidth=2)

        su = regret_runs[regret_runs['tag'] == row['tag']]['TS_up_wc'].tolist()
        ci_u = sms.DescrStatsW(su).tconfint_mean()
        l_u = abs(ci_u[1]-ci_u[0])
        plt.errorbar(row['TS_up_wc'], ix-.2, xerr=l_u, color='b', fmt='none', capsize=5, linewidth=2, elinewidth=2)

        plt.plot([row['R'], row['R']], [ix-.28, ix+.33], color='k', linewidth=2, zorder=3)

    # Create a custom legend
    custom_legend = [
        plt.Line2D([0], [0], color='r', lw=0, label='Ours'),
        plt.Line2D([0], [0], color='r', lw=2, label='One step (IV)'),
        plt.Line2D([0], [0], color='r', lw=2, label='One step (WC)', linestyle='--'),
        plt.Line2D([0], [0], color='r', lw=0, label=''),
        plt.Line2D([-0.2], [0], color='r', lw=0, label='Baseline'),
        plt.Line2D([0], [0], color='b', lw=2, label='Two step (IV)'),
        plt.Line2D([0], [0], color='b', lw=2, label='Two step (WC)', linestyle='--'),
        plt.Line2D([0], [0], color='r', lw=0, label=''),
        plt.Line2D([0], [0], color='k', marker='s', markersize=10, label='Oracle regret'),
    ]

    lgd = plt.legend(handles=custom_legend,  fontsize=15, bbox_to_anchor=(1.3, 1.03))

    for t in lgd.get_texts():
        t.set_ha('left')

    plt.xlabel(f'{metric} Regret')
    keys = [i for i in range(regret_df.shape[0])]
    vals = [b + f' ($\gamma={a:.2}, \omega={c:.2}$)' for a,b,c in zip(regret_df['SR'].to_list(), regret_df['tag'].to_list(), regret_df['pG'].to_list())]
    plt.yticks(keys, vals)
    
    if save:
        plt.savefig('bound_plot.pdf', dpi=500, bbox_inches='tight')


def plot_regret_seperation(bdf):
    
    plt.figure()
    bkeys = ['D_y0', 'D_y1', 'D_a0', 'D_u']
    metrics = ['m_y=0', 'm_y=1', 'm_a=0', 'm_u']
    metric_names = ['FPR', 'TPR', 'NPV', 'Accuracy']
    xnames = ['$\overline{\Delta}(m_{y=0})$', '$\overline{\Delta}(m_{y=1})$',
            '$\overline{\Delta}(m_{a=0})$', '$\overline{\Delta}(m_u)$']

    n_plots = len(metrics)
    n_bins = 9
    fig, axs = plt.subplots(1, n_plots, figsize=(5*n_plots, 5), sharey=False)

    for ix, bkey in enumerate(bkeys):

        mdf = bdf[bdf['metric'] == metrics[ix]]
        mdf[bkey] = ((mdf[bkey]*n_bins).round()/n_bins).copy()
        mdf = mdf.groupby([bkey]).mean().reset_index()
        mdf = mdf.sort_values(by=bkey, ascending=True)
        
        axs[ix].fill_between(mdf[bkey], mdf['Rs_down'],mdf['Rs_up'], color='#708090', alpha=.5, label='$R$')
        axs[ix].fill_between(mdf[bkey], mdf['Rd_down'], mdf['Rd_up'], color='#F5DEB3', alpha=.5, label='$R_{\delta}$')

        axs[ix].plot(mdf[bkey], mdf[f'R_oracle'], color='k', label='$R^*$')
        axs[ix].set_title(metric_names[ix], fontsize=18)
        axs[ix].set_xlabel(xnames[ix], fontsize=18)
        axs[ix].axhline(color='grey', alpha=.2)

    axs[0].set_ylabel('Regret', fontsize=18)
    plt.legend(fontsize=15)
    plt.savefig('figs/seperation.pdf', dpi=500, bbox_inches='tight')


def plot_util_regret_seperation(bdf):
    
    plt.figure()
    udf = bdf[(bdf['metric'] == 'm_u') & (bdf['u_01'] == 0) & (bdf['u_10'] == 0)]

    udf['ur'] = udf['u_00'] / udf['u_11']
    udf = udf.sort_values(by='D_u')
    udf = udf.groupby('D_u').mean().reset_index()

    plt.axhline(0, color='grey', zorder=1, linestyle='--')

    plt.fill_between(udf['D_u'], udf['Rs_down'], udf['Rs_up'], color='#708090', alpha=.5, label='$R$')
    plt.fill_between(udf['D_u'], udf['Rd_down'], udf['Rd_up'], color='#F5DEB3', alpha=.5, label='$R_{\delta}$')
    plt.plot(udf['D_u'], udf['R_oracle'], color='k', label='$R^*$')
    plt.xscale('log',base=10) 
    plt.legend(loc='lower right', fontsize=12)


    plt.xlabel(r'$\beta$', fontsize=14)
    plt.ylabel('Regret', fontsize=14)
    plt.savefig('figs/seperation_utility.pdf', dpi=500, bbox_inches='tight')


def plot_exclusion_sensitivity(bedf, path):

    metric_dict = {
        'm_y=1': 'TPR',
        'm_y=0': 'FPR',
        'm_a=0': 'NPV',
        'm_a=1': 'PPV',
        'm_u': 'ACCURACY',
    }

    metrics = bedf['metric'].unique().tolist()

    fig, axes = plt.subplots(1, 5, figsize=(25, 5)) # Adjust the figsize as needed

    for i, metric in enumerate(metrics):
        # Filter the DataFrame for the current metric
        metric_df = bedf[bedf['metric'] == metric]
        
        # Reset the index to avoid the duplicate labels error
        metric_df = metric_df.reset_index(drop=True)
        
        # Create each lineplot on the corresponding subplot axis
        sns.lineplot(ax=axes[i], data=metric_df, x='beta_zy', y='Rs_up', color='orange', linestyle='-', label='Baseline')
        sns.lineplot(ax=axes[i], data=metric_df, x='beta_zy', y='Rd_up', color='blue', linestyle='--', label='Delta (ours)')
        sns.lineplot(ax=axes[i], data=metric_df, x='beta_zy', y='Rs_down', color='orange', linestyle='-')
        sns.lineplot(ax=axes[i], data=metric_df, x='beta_zy', y='Rd_down', color='blue', linestyle='--')
        sns.lineplot(ax=axes[i], data=metric_df, x='beta_zy', y='R_oracle', color='black', label='Oralce')

        # Set the titles, labels, etc.
        axes[i].set_xlabel(r'Exclusion violation ($\beta_1$)', fontsize=18)
        axes[i].set_title(f'{metric_dict[metric]}', fontsize=20)
        axes[i].set_ylabel(f'', fontsize=18)
        axes[i].legend().set_visible(False)
        
    axes[0].set_ylabel(f'Regret', fontsize=20)
    axes[0].legend(fontsize=16)

    plt.savefig(path, dpi=500, bbox_inches='tight')


def plot_relevance_sensitivity(brdf, path):

    metric_dict = {
        'm_y=1': 'TPR',
        'm_y=0': 'FPR',
        'm_a=0': 'NPV',
        'm_a=1': 'PPV',
        'm_u': 'ACCURACY',
    }

    metrics = brdf['metric'].unique().tolist()

    fig, axes = plt.subplots(1, 5, figsize=(25, 5)) # Adjust the figsize as needed

    for i, metric in enumerate(metrics):
        
        # Filter the DataFrame for the current metric
        metric_df = brdf[brdf['metric'] == metric]
        
        # Reset the index to avoid the duplicate labels error
        metric_df = metric_df.reset_index(drop=True)
        
        # Create each lineplot on the corresponding subplot axis
        sns.lineplot(ax=axes[i], data=metric_df, x='beta_zy', y='Rs_up', color='orange', linestyle='-', label='Baseline')
        sns.lineplot(ax=axes[i], data=metric_df, x='beta_zy', y='Rs_down', color='orange', linestyle='-')
        sns.lineplot(ax=axes[i], data=metric_df, x='beta_zy', y='Rd_up', color='blue', linestyle='--', label='Delta (ours)')
        sns.lineplot(ax=axes[i], data=metric_df, x='beta_zy', y='Rd_down', color='blue', linestyle='--')
        sns.lineplot(ax=axes[i], data=metric_df, x='beta_zy', y='R_oracle', color='black', label='Oracle')

        # Set the titles, labels, etc.
        axes[i].set_xlabel(r'IV relevance ($\beta_0$)', fontsize=16)
        axes[i].set_title(f'{metric_dict[metric]}', fontsize=20)
        axes[i].set_ylabel(f'', fontsize=18)
        axes[i].legend().set_visible(False)
        
    axes[0].set_ylabel(f'Regret', fontsize=20)
    axes[0].legend(fontsize=16)
    plt.savefig(path, dpi=500, bbox_inches='tight')

def plot_design_sensitivity(brdf, path):

    metric_dict = {
        'm_y=1': 'TPR',
        'm_y=0': 'FPR',
        'm_a=0': 'NPV',
        'm_a=1': 'PPV',
        'm_u': 'ACCURACY',
    }
    nplots = len(metric_dict.keys())

    fig, axes = plt.subplots(1, nplots, figsize=(5*nplots, 5)) # Adjust the figsize as needed
    metrics = brdf['metric'].unique().tolist()

    for i, metric in enumerate(metric_dict.keys()):

        # Filter the DataFrame for the current metric
        mdf = brdf[brdf['metric'] == metric]

        # Reset the index to avoid the duplicate labels error
        mdf = mdf.reset_index(drop=True)

        mdf['Rs_includes_zero'] = ((mdf['Rs_down'] <= 0) & (mdf['Rs_up'] >= 0)).astype(int)
        mdf['Rd_includes_zero'] = ((mdf['Rd_down'] <= 0) & (mdf['Rd_up'] >= 0)).astype(int)
        TS_lambda_star = mdf[mdf['Rs_includes_zero'] == 1]['lambda'].min()
        OS_lambda_star = mdf[mdf['Rd_includes_zero'] == 1]['lambda'].min()

        axes[i].fill_between(mdf['lambda'], mdf['Rs_down'], mdf['Rs_up'], label='Baseline interval', color='#a8a8a8', alpha=.5, )
        axes[i].fill_between(mdf['lambda'], mdf['Rd_down'], mdf['Rd_up'], label='Delta interval (ours)', color='#1E90FF', alpha=.5,)

        regret_intercept = mdf['Rs_down'].tolist()[0]
        if 'y' in metric:
            axes[i].axvline(TS_lambda_star, color='#708090', zorder=1, linestyle='--')
            axes[i].axvline(OS_lambda_star, color='#708090', zorder=1, linestyle='--')
            mval = mdf[['Rs_down', 'Rd_down']].min().min()
            axes[i].text(TS_lambda_star+.05, mval, '$\Lambda^{0}_{R}$', fontsize=18, color='black')
            axes[i].text(OS_lambda_star-.05, mval-.005, '$\Lambda^{0}_{R_\delta}$', fontsize=18, color='black')

        axes[i].axhline(0, color='grey', zorder=1, linestyle='--')
        axes[i].set_title(f'{utils.metric_dict[metric]}', fontsize=16)
        axes[i].set_xlabel('$\Lambda$', fontsize=16)

    axes[-1].legend(loc='upper right', fontsize=16)    
    axes[0].set_ylabel('Regret', fontsize=16)

    plt.savefig(path, dpi=500, bbox_inches='tight')


def plot_cost_ratio_curve(dgp, crdf, fname):

    crdf = crdf.sort_values(by='cr')
    crdf = crdf.groupby('cr').mean().reset_index()

    plt.axhline(0, color='grey', zorder=1, linestyle='--')
    plt.fill_between(crdf['cr'], crdf['Rs_down'],crdf['Rs_up'], alpha=.5, label='Baseline interval', color='#a8a8a8')
    plt.fill_between(crdf['cr'], crdf['Rd_down'],crdf['Rd_up'], alpha=.5, label='Delta interval (ours)', color='#1E90FF')
    plt.xscale('log',base=10) 
    plt.legend(loc='upper right')

    plt.xlabel('False Positive vs. False Negative Cost Ratio', fontsize=12)
    plt.ylabel('Cost Regret', fontsize=12)
    plt.savefig(fname, dpi=500)


def plot_msm_sensitivity(msm_dgp, msmdf, path):

    metric_dict = {
        'm_y=1': 'TPR',
        'm_y=0': 'FPR',
        'm_a=0': 'NPV',
        'm_a=1': 'PPV',
        'm_u': 'ACCURACY',
    }

    lam = msm_dgp['lambda']
    metrics = msmdf['metric'].unique().tolist()

    fig, axes = plt.subplots(1, 5, figsize=(25, 5)) # Adjust the figsize as needed

    for i, metric in enumerate(metrics):

        # Filter the DataFrame for the current metric
        metric_df = msmdf[msmdf['metric'] == metric]

        # Reset the index to avoid the duplicate labels error
        metric_df = metric_df.reset_index(drop=True)
        ymin = metric_df[['Rs_down', 'Rs_up', 'Rd_down', 'Rd_up', 'R_oracle']].min().min()
        ymax = metric_df[['Rs_down', 'Rs_up', 'Rd_down', 'Rd_up', 'R_oracle']].max().max()

        # Create each lineplot on the corresponding subplot axis
        sns.lineplot(ax=axes[i], data=metric_df, x='ls', y='Rs_up', color='orange', linestyle='-', label='Baseline')
        sns.lineplot(ax=axes[i], data=metric_df, x='ls', y='Rs_down', color='orange', linestyle='-')
        sns.lineplot(ax=axes[i], data=metric_df, x='ls', y='Rd_up', color='b', linestyle='--', label='Delta (ours)')
        sns.lineplot(ax=axes[i], data=metric_df, x='ls', y='Rd_down', color='b', linestyle='--')
        sns.lineplot(ax=axes[i], data=metric_df, x='ls', y='R_oracle', color='black', label='Oracle')

        # Set the titles, labels, etc.
        axes[i].set_xlabel(r'$\Lambda^*$', fontsize=20)
        axes[i].set_title(f'{metric_dict[metric]}', fontsize=20)
        axes[i].set_ylabel(f'', fontsize=16)

        ymin, ymax = axes[i].get_ylim()
        axes[i].fill_between([lam**-1, lam], ymin, ymax, color='grey',
                             alpha=0.2, zorder=-1, label='Coverage')
        axes[i].legend().set_visible(False)

    axes[0].set_ylabel(f'Regret', fontsize=20)
    axes[0].legend(fontsize=16)
    plt.savefig(path, dpi=500, bbox_inches='tight')

def plot_subgroup_basic(gbdf, metric, fname):
    
    mdf = gbdf[gbdf['metric'] == metric].reset_index(drop=True)[::-1]
    groups = mdf['g'].to_list()

    plt.axvline(0, color='grey', zorder=1, linestyle='--')
    gdfm = mdf.groupby(['g']).mean().reset_index()[::-1].reset_index()

    for ix, row in gdfm.iterrows():

        gdf = mdf[mdf['g'] == row['g']]
        sd = gdf['Rs_down'].tolist()
        ci_d = sms.DescrStatsW(sd).tconfint_mean()
        l_d = abs(ci_d[1]-ci_d[0])

        sd = gdf['Rs_up'].tolist()
        ci_d = sms.DescrStatsW(sd).tconfint_mean()
        l_u = abs(ci_d[1]-ci_d[0])
        
        plt.scatter(row['Rs_down'], ix,  color='r', s=30)
        plt.scatter(row['Rs_up'], ix, color='r', label="R")
        plt.plot([row['Rs_down'], row['Rs_up']], [ix, ix], color='r', label="Baseline interval", linewidth=2)

        sd = gdf['Rd_down'].tolist()
        ci_d = sms.DescrStatsW(sd).tconfint_mean()
        l_d = abs(ci_d[1]-ci_d[0])

        sd = gdf['Rd_up'].tolist()
        ci_d = sms.DescrStatsW(sd).tconfint_mean()
        l_u = abs(ci_d[1]-ci_d[0])
        
        plt.plot([row['Rd_down'], row['Rd_up']], [ix+.15, ix+.15], color='b', label="Delta interval (ours)", linewidth=2)
        plt.scatter(row['Rd_down'], ix+.15,  color='b', s=30)
        plt.scatter(row['Rd_up'], ix+.15, color='b', label="R")


    keys = [i for i in range(gdfm.shape[0])]
    vals = realdata.get_group_descs(gdfm, sr_info=True)
    plt.yticks(keys, vals, fontsize=12)

    plt.xlabel(f'{utils.metric_dict[metric]} Regret', fontsize=12)

    # Create a custom legend
    custom_legend = [
        plt.Line2D([0], [0], color='b', lw=2, label='Delta interval (ours)', linestyle='-'),
        plt.Line2D([0], [0], color='r', lw=2, label='Baseline interval', linestyle='-'),
    ]

    lgd = plt.legend(handles=custom_legend,  fontsize=10, loc='lower left')
    plt.savefig(fname, dpi=500, bbox_inches='tight')
