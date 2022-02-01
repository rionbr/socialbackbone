# coding=utf-8
# Author: Rion B Correia
# Date: Dec 05, 2017
#
# Description:
# Plot Neighbour distributions from Networks (calculated by calc_neighbours.py)
#
from __future__ import division
# Plotting
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from matplotlib.lines import Line2D
# General
import pandas as pd
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
# Networks
import utils


def plot_neighbours(source, project, normalization, plottypes, date=None):

    print('--- Ploting: %s : %s : %s : %s ---' % (source, project, normalization, date))
    loglog = plottypes['loglog']

    # Init
    if date is not None:
        rCSVFile = 'results/%s/%s/%s/neighbours-%s.csv' % (source, project, date, normalization)
        project_str = project.replace('-', ' ').title() + ' ' + date

    else:
        rCSVFile = 'results/%s/%s/neighbours-%s.csv' % (source, project, normalization)
        project_str = project.replace('-', ' ').title()

    source_str = source.title()

    # Load DataFrame
    print('> Loading DataFrame')
    df = pd.read_csv(rCSVFile, header=0, index_col=0, encoding='utf-8').set_index('edges', drop=True)

    if normalization == 'social':
        normalization_str = "Social"
    elif normalization == 'time':
        normalization_str = "Individual time"
    elif normalization == 'time_all':
        normalization_str = "Experiment time"

    if date is not None:
        wImgFile = "images/%s/%s/%s/%s/neighbours.pdf" % (
            source, project, date, normalization
        )
    else:
        wImgFile = "images/%s/%s/%s/neighbours.pdf" % (
            source, project, normalization
        )

    title = u'Same module neighbors\n%s-%s-%s' % (source_str, project_str, normalization_str)
    print(df.head())

    print('--- Plotting ---')

    fig, (ax) = plt.subplots(figsize=(5, 4), nrows=1, ncols=1)
    axi = zoomed_inset_axes(ax, zoom=3, loc='upper center')

    # plt.rc('font', size=10)
    # plt.rc('legend', fontsize=12)
    plt.rc('legend', numpoints=1)
    # plt.rc('axes', titlesize=11)
    # plt.rc('axes', labelsize=14)

    lw = 2
    ms = 8

    # Original
    color_o = 'orange'
    markeredgecolor_o = 'dark' + color_o
    ls_o = 'dashed'
    marker_o = 'D'
    plot_o, = ax.plot(df.index.values, df['ks-original'].values,
                      color=color_o, markerfacecolor=color_o, markeredgecolor=markeredgecolor_o,
                      alpha=1.0, linestyle=ls_o, lw=lw, marker=None, ms=ms, zorder=8, label=r'Original'
                      )
    plot_o_i, = axi.plot(df.index.values, df['ks-original'].values,
                         color=color_o, markerfacecolor=color_o, markeredgecolor=markeredgecolor_o,
                         alpha=1.0, linestyle=ls_o, lw=(lw / 2), marker=marker_o, ms=(ms / 2), zorder=8, label=r'Original'
                         )

    # Metric Backbone
    color_b = 'red'
    markeredgecolor_b = 'dark' + color_b
    ls_b = '-'
    marker_b = 'o'
    plot_b, = ax.plot(df.index.values, df['ks-metric'].values,
                      color=color_b, markerfacecolor=color_b, markeredgecolor=markeredgecolor_b,
                      alpha=1.0, linestyle=ls_b, lw=lw, marker=None, ms=ms, zorder=10, label=r'Metric'
                      )
    plot_b_i, = axi.plot(df.index.values, df['ks-metric'].values,
                         color=color_b, markerfacecolor=color_b, markeredgecolor=markeredgecolor_b,
                         alpha=1.0, linestyle=ls_b, lw=(lw / 2), marker=marker_b, ms=(ms / 2), zorder=10, label=r'Metric'
                         )

    # Threshold
    color_t = 'blue'
    markeredgecolor_t = 'dark' + color_t
    ls_t = 'dashdot'
    marker_t = '^'
    plot_t, = ax.plot(df.index.values, df['ks-threshold'].values,
                      color=color_t, markerfacecolor=color_t, markeredgecolor=markeredgecolor_t,
                      alpha=1.0, linestyle=ls_t, lw=lw, marker=None, ms=ms, zorder=6, label=r'Threshold'
                      )
    plot_t_i, = axi.plot(df.index.values, df['ks-threshold'].values,
                         color=color_t, markerfacecolor=color_t, markeredgecolor=markeredgecolor_t,
                         alpha=1.0, linestyle=ls_t, lw=(lw / 2), marker=marker_t, ms=(ms / 2), zorder=6, label=r'Threshold'
                         )

    # Random
    color_r = 'green'
    markeredgecolor_r = 'dark' + color_r
    ls_r = 'dotted'
    marker_r = 'v'
    plot_r, = ax.plot(df.index.values, df['ks-random'].values,
                      color=color_r, markerfacecolor=color_r, markeredgecolor=markeredgecolor_r,
                      alpha=1.0, linestyle=ls_r, lw=lw, marker=None, ms=ms, zorder=4, label=r'Random'
                      )
    plot_r_i, = axi.plot(df.index.values, df['ks-random'].values,
                         color=color_r, markerfacecolor=color_r, markeredgecolor=markeredgecolor_r,
                         alpha=1.0, linestyle=ls_r, lw=(lw / 2), marker=marker_r, ms=(ms / 2), zorder=4, label=r'Random'
                         )

    #
    # sub region of the original image
    #
    xmin, xmax = 0, 10
    ymin, ymax = df.max().max() - 0.2, df.max().max()
    axi.set_xlim(xmin, xmax)
    axi.set_ylim(ymin, ymax)
    # fix the number of ticks on the inset axes
    # axi.yaxis.get_major_locator().set_params(nbins=7)
    # axi.xaxis.get_major_locator().set_params(nbins=7)

    axi.tick_params(axis='both', which='both', bottom=False, left=False, labelleft=False, labelbottom=False)
    # axi.set_yticks([])
    axi.grid()
    mark_inset(ax, axi, loc1=2, loc2=3, fc="none", ec="0.5")

    #
    # Plot definitions
    #
    ax.set_title(title)
    ax.set_ylabel(r'$\Lambda$', fontsize='large')
    ax.set_xlabel(r'K', fontsize='large')

    if (loglog):
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        pass
    ax.grid()

    #
    # Custom Legend
    #
    ax.legend(
        [
            Line2D([], [], color=color_o, marker=marker_o, markerfacecolor=color_o, markeredgecolor=markeredgecolor_o, lw=lw, ls=ls_o),
            Line2D([], [], color=color_b, marker=marker_b, markerfacecolor=color_b, markeredgecolor=markeredgecolor_b, lw=lw, ls=ls_b),
            Line2D([], [], color=color_t, marker=marker_t, markerfacecolor=color_t, markeredgecolor=markeredgecolor_t, lw=lw, ls=ls_t),
            Line2D([], [], color=color_r, marker=marker_r, markerfacecolor=color_r, markeredgecolor=markeredgecolor_r, lw=lw, ls=ls_r),
        ],
        [
            'Original',
            'Metric',
            'Threshold',
            'Random',
        ],
        loc='upper right', borderpad=1.00, ncol=1, columnspacing=0.5  # , fontsize=10,
    )

    utils.ensurePathExists(wImgFile)
    # plt.tight_layout()
    plt.savefig(wImgFile, dpi=150, frameon=True, bbox_inches='tight', pad_inches=0.05)
    plt.close()


if __name__ == '__main__':
    #
    # Init
    #
    source = 'sociopatterns'  # sociopatterns, salanthe, toth
    # # salanthe: high-school
    # # sociopatterns: high-school, hospital, primary-school, workplace
    # # toth: elementary-school, middle-school
    project = 'workplace'
    normalization = 'social'  # social, time, time_all
    # time_window = '20S'

    # # For Toth projects
    # # - elementary-school = ['2013-01-31','2013-02-01','all']
    # # - middle-school = :['2012-11-28','2012-11-29','all']
    # date = 'all'
    date = None

    plottypes = {'loglog': True}

    # Social, Individual Time, Experiment Time

    if project == 'exhibit':
        # start='2009-04-28', end='2009-07-17'
        date_range = pd.date_range(start='2009-07-15', end='2009-07-15')
        for date in date_range:
            datestr = date.strftime('%Y-%m-%d')

            # Some dates do not exist, just skip
            if datestr in ['2009-05-04', '2009-05-08', '2009-05-11', '2009-05-18', '2009-05-25', '2009-06-01',
                           '2009-06-08', '2009-06-15', '2009-06-22', '2009-06-29', '2009-07-06', '2009-07-13']:
                continue
            plot_neighbours(source, project, normalization, plottypes, date=datestr)

    else:
        plot_neighbours(source, project, normalization, plottypes, date=date)
