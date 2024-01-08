import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.stats.api as sms


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