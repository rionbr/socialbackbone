# coding=utf-8
# Author: Rion B Correia
# Date: Dec 05, 2017
#
# Description:
# Plot Epidemic Result from Alain (results_epidemic)
#
from __future__ import division
# Plotting
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['xtick.labelsize'] = 'medium'
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
# General
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
# Networks
import utils


def plot_epidemics_result(source, project, normalization, plottypes, date=None, *args, **kwargs):

    print('--- Ploting: {:} : {:} : {:} : {:} : ---'.format(source, project, normalization, plottypes))
    loglog = plottypes['loglog']
    datatype = plottypes['datatype']
    showTitle = kwargs.get('showTitle', True)

    # Init
    if date is not None:
        rTauFile = 'results_epidemic/%s/%s/tau_vs_x_%s-%s_%s.dat' % (source, project, project, date, datatype)
        rTauRndFile = 'results_epidemic/%s/%s/tau_rnd_vs_x_%s-%s_%s.dat' % (source, project, project, date, datatype)
        rTauLargeWFile = 'results_epidemic/%s/%s/tau_largew_vs_x_%s-%s_%s.dat' % (source, project, project, date, datatype)
        project_str = project.replace('-', ' ').title() + ' - ' + date.title()
    else:
        rTauFile = 'results_epidemic/%s/%s/tau_vs_x_%s_%s.dat' % (source, project, project, datatype)
        rTauRndFile = 'results_epidemic/%s/%s/tau_rnd_vs_x_%s_%s.dat' % (source, project, project, datatype)
        rTauLargeWFile = 'results_epidemic/%s/%s/tau_largew_vs_x_%s_%s.dat' % (source, project, project, datatype)
        project_str = project.replace('-', ' ').title()

    # Load .dat files
    df = pd.read_csv(rTauFile, sep='\t', header=None, names=['edges', 'tau_1/2', 'tau_1'])
    dfR = pd.read_csv(rTauRndFile, sep='\t', header=None, names=['edges', 'tau_1/2', 'tau_1', 'tau_1/2-old', 'tau_1-old', 'frac_conn', 'avg_size_non-conn'])
    dfT = pd.read_csv(rTauLargeWFile, sep='\t', header=None, names=['edges', 'tau_1/2', 'tau_1', 'tau_1/2-old', 'tau_1-old', 'frac_conn', 'avg_size_non-conn'])

    # Zero to NaN
    dfR['tau_1/2'] = dfR['tau_1/2'].replace({0: np.nan})
    dfR['tau_1'] = dfR['tau_1'].replace({0: np.nan})
    dfT['tau_1/2'] = dfT['tau_1/2'].replace({0: np.nan})
    dfT['tau_1'] = dfT['tau_1'].replace({0: np.nan})

    source_str = source.title()

    # Load Network
    if normalization == 'social':
        normalization_str = "social"
    elif normalization == 'time':
        normalization_str = "Ind. time"
    elif normalization == 'time_all':
        normalization_str = "Exp. time"

    if date is not None:
        wImgFile = "images/%s/%s/%s/%s/epidemic-taus.pdf" % (source, project, date, normalization)
    else:
        wImgFile = "images/%s/%s/%s/epidemic-taus.pdf" % (source, project, normalization)

    print('--- Plotting ---')
    fig, (ax) = plt.subplots(figsize=(5.8, 3.9), nrows=1, ncols=1)
    ax2 = ax.twinx()
    ax.patch.set_visible(False)
    ax2.patch.set_visible(True)
    ax.set_zorder(ax2.get_zorder() + 1)

    # plt.rc('legend', fontsize=11)
    # plt.rc('font', size=10)
    # plt.rc('legend', numpoints=1)
    # plt.rc('axes', labelsize=300)
    lw = 2
    ms = 8

    # Metric Backbone
    # Tau vs X (1-backbone (nb edges) + random edges)
    markerfacecolor_b = '#ff9896'  # red
    color_b = markeredgecolor_b = '#d62728'  # 'dark' + color_b
    ls_b = '-'
    marker_b = 'o'
    plot_b_full, = ax.plot(df.index.values, df['tau_1'].values,
                           color=color_b, markerfacecolor=markerfacecolor_b, markeredgecolor=markeredgecolor_b,
                           alpha=1.0, linestyle=ls_b, lw=lw, marker=marker_b, ms=ms,
                           rasterized=False, zorder=8,
                           label=r'$\tau_{1}$, Metric backbone'
                           )
    plot_b_half, = ax.plot(df.index.values, df['tau_1/2'].values,
                           color=color_b, markerfacecolor=markerfacecolor_b, markeredgecolor=markeredgecolor_b,
                           alpha=1.0, linestyle=ls_b, lw=lw, marker=marker_b, ms=(ms / 2),
                           rasterized=False, zorder=8,
                           label=r'Metric backbone, $\tau_{1/2}$'
                           )

    # Threshold Backbone
    # Tau LargeW vs X (3-subgraph of the nb edges with largest weights + random edges)
    markerfacecolor_t = '#aec7e8'  # blue
    color_t = markeredgecolor_t = '#1f77b4'  # 'dark' + color_t
    ls_t = 'dashdot'
    marker_t = '^'
    plot_t_full, = ax.plot(dfT.index.values, dfT['tau_1'].values,
                           color=color_t, markerfacecolor=markerfacecolor_t, markeredgecolor=markeredgecolor_t,
                           alpha=1.0, linestyle=ls_t, lw=lw, marker=marker_t, ms=ms,
                           rasterized=False, zorder=6,
                           label=r'$t_{1}$, Threshold backbone'
                           )
    plot_t_half, = ax.plot(dfT.index.values, dfT['tau_1/2'].values,
                           color=color_t, markerfacecolor=markerfacecolor_t, markeredgecolor=markeredgecolor_t,
                           alpha=1.0, linestyle=ls_t, lw=lw, marker=marker_t, ms=(ms / 2),
                           rasterized=False, zorder=6,
                           label=r'$t_{1/2}$, Threshold backbone'
                           )

    # Random Backbone
    # Tau (rnd) vs X (2-random subgraph with same number of edges than backbone + random edges)
    markerfacecolor_r = '#98df8a'  # green
    color_r = markeredgecolor_r = '#2ca02c'  # 'dark' + color_r
    ls_r = 'dotted'
    marker_r = 'v'
    plot_r_full, = ax.plot(dfR.index.values, dfR['tau_1'].values,
                           color=color_r, markerfacecolor=markerfacecolor_r, markeredgecolor=markeredgecolor_r,
                           alpha=1.0, linestyle=ls_r, lw=lw, marker=marker_r, ms=ms,
                           rasterized=False, zorder=4,
                           label=r'$t_{1}$, Random backbone'
                           )
    plot_r_half, = ax.plot(dfR.index.values, dfR['tau_1/2'].values,
                           color=color_r, markerfacecolor=markerfacecolor_r, markeredgecolor=markeredgecolor_r,
                           alpha=1.0, linestyle=ls_r, lw=lw, marker=marker_r, ms=(ms / 2),
                           rasterized=False, zorder=4,
                           label=r'$t_{1/2}$, Random backbone'
                           )

    # Annotate fraction of networks which are connected
    ax2.bar(x=(dfR.index.values - 0.1), height=dfR['frac_conn'], width=0.22, color='#2ca02c', alpha=0.2, zorder=2)
    ax2.bar(x=(dfT.index.values + 0.1), height=dfT['frac_conn'], width=0.22, color='#1f77b4', alpha=0.2, zorder=2)
    ax2.set_ylabel('Fraction of Connected Networks')

    #
    # Plot definitions
    #
    if showTitle:
        title = u'Time to infection on subgraphs\n%s - %s (%s)' % (source_str, project_str, normalization_str)
        ax.set_title(title)

    ax.set_ylabel(r'$t$ $(\mathrm{partial})$ $/$ $t$ $(\mathrm{full \; network})$', fontsize='large')
    #ax.set_xlabel(r'Percentage of edges')
    ax.set_xlabel(r'$\chi$', fontsize='large')

    ax.set_xticks(dfR.index.values)
    ax.set_xticklabels([r'$|B(X)|$'] + ['{:d}%'.format(x) for x in range(10, 100, 10)] + [r'$|D(X)|$'])

    ax2.set_yticks(np.linspace(0, 1, 5))
    ax2.set_yticklabels(np.linspace(0, 1, 5))

    ax.set_xlim(-0.3, 10.3)

    if (loglog):
        ax.set_yscale("log")
    ax.grid(axis='x', linestyle='-')
    ax.grid(axis='y', linestyle='-.')
    ax2.grid(axis='y', linestyle=':')

    #
    # Custom Legend
    #
    big_markers = (
        Line2D([], [], color=color_b, lw=0, marker=marker_b, ms=ms, markerfacecolor=markerfacecolor_b, markeredgecolor=markeredgecolor_b),
        Line2D([], [], color=color_t, lw=0, marker=marker_t, ms=ms, markerfacecolor=markerfacecolor_t, markeredgecolor=markeredgecolor_t),
        Line2D([], [], color=color_r, lw=0, marker=marker_r, ms=ms, markerfacecolor=markerfacecolor_r, markeredgecolor=markeredgecolor_r)
    )
    small_markers = (
        Line2D([], [], color=color_b, lw=0, marker=marker_b, ms=(ms / 2), markerfacecolor=markerfacecolor_b, markeredgecolor=markeredgecolor_b),
        Line2D([], [], color=color_t, lw=0, marker=marker_t, ms=(ms / 2), markerfacecolor=markerfacecolor_t, markeredgecolor=markeredgecolor_t),
        Line2D([], [], color=color_r, lw=0, marker=marker_r, ms=(ms / 2), markerfacecolor=markerfacecolor_r, markeredgecolor=markeredgecolor_r)
    )
    ax.legend(
        [
            big_markers,
            small_markers,
            Line2D([], [], color=(0, 0, 0, 0)),  # blank
            Line2D([], [], color=color_b, lw=lw, ls=ls_b),
            Line2D([], [], color=color_t, lw=lw, ls=ls_t),
            Line2D([], [], color=color_r, lw=lw, ls=ls_r),
        ],
        [
            r'$t_{1}$',
            r'$t_{1/2}$',
            ' ',
            'Backbone',
            'Threshold',
            'Random',
        ],
        handler_map={
            big_markers: HandlerTuple(ndivide=None),
            small_markers: HandlerTuple(ndivide=None),
        },
        loc='upper right', borderpad=0.40, ncol=2, columnspacing=0.5,
    )

    print('file: {:s}'.format(wImgFile))
    utils.ensurePathExists(wImgFile)
    #plt.tight_layout()
    plt.subplots_adjust(left=0.11, right=0.88, bottom=0.14, top=0.86, wspace=0, hspace=0.0)
    plt.savefig(wImgFile, dpi=150)  # , pad_inches=0.05)
    plt.close()


if __name__ == '__main__':
    #
    # Init
    #
    source = 'toth'  # sociopatterns, salathe, toth
    # # salathe: high-school
    # # sociopatterns: exhibit, high-school, hospital, conference, primary-school, workplace
    # # toth: elementary-school, middle-school
    project = 'middle-school'
    normalization = 'social'  # social, time, time_all
    # time_window = '20S'

    # # For Toth projects
    # # - elementary-school = ['2013-01-31','2013-02-01','all']
    # # - middle-school = :['2012-11-28','2012-11-29','all']
    date = 'all'
    #date = None

    # Main paper figure has no title
    showTitle = True

    plottypes = {'datatype': 'ED', 'loglog': True}

    """
    if project == 'exhibit':
        # start='2009-04-28', end='2009-07-17'
        date_range = pd.date_range(start='2009-07-15', end='2009-07-15')
        for date in date_range:
            datestr = date.strftime('%Y-%m-%d')

            # Some dates do not exist, just skip
            if datestr in ['2009-05-04','2009-05-08','2009-05-11','2009-05-18','2009-05-25','2009-06-01',
                '2009-06-08','2009-06-15','2009-06-22','2009-06-29','2009-07-06','2009-07-13']:
                continue
            plot_distributions(source, project, normalization, time_window, plottypes, date=datestr)
    else:
        plot_distributions(source, project, normalization, time_window, plottypes, date=date)
        """
    plot_epidemics_result(source, project, normalization, plottypes, date, showTitle=showTitle)
