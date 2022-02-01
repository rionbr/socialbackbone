# coding=utf-8
# Author: Rion B Correia
# Date: Dec 05, 2017
#
# Description:
# Plot S_{ij} value distribution for networks
#
from __future__ import division
# Plotting
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
mpl.rcParams['xtick.labelsize'] = 'medium'
import matplotlib.pyplot as plt
# General
import pandas as pd
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
# Networks
import networkx as nx
import powerlaw


def generate_original_graph(G):
    GO = G.copy()
    edges2remove = [(i, j) for i, j, d in G.edges(data=True) if 'original' not in d]
    GO.remove_edges_from(edges2remove)
    return GO


def plot_s_dist(source, project, normalization, date=None):
    # Init
    if date is not None:
        rGpickle = 'results/%s/%s/%s/graph-%s.gpickle' % (source, project, date, normalization)
    else:
        rGpickle = 'results/%s/%s/graph-%s.gpickle' % (source, project, normalization)

    source_str = source.title()

    if date is not None:
        wImgFile = "images/%s/%s/%s/%s/dist-s-values.pdf" % (source, project, date, normalization)
        project_str = project.replace('-', ' ').title() + ' - ' + date.title()
    else:
        wImgFile = "images/%s/%s/%s/dist-s-values.pdf" % (source, project, normalization)
        project_str = project.replace('-', ' ').title()

    # Load Network
    print('--- Loading Network gPickle (%s) ---' % (rGpickle))
    G = nx.read_gpickle(rGpickle)
    print('Network: %s' % (G.name))
    #
    # SubGraphs
    #
    # Original Graph, without the metric closure
    GO = generate_original_graph(G)

    ss = pd.Series([d.get('s_value') for i, j, d in GO.edges(data=True)], name='s-value')

    # Select only s-values
    dfs = ss.loc[(ss > 1.0)].sort_values(ascending=False).to_frame()
    xmin = dfs['s-value'].min()
    xmin = 1
    fit = powerlaw.Fit(dfs['s-value'], xmin=xmin, estimate_discrete=False)

    alpha = fit.power_law.alpha
    sigma = fit.power_law.sigma
    print('Powerlaw: alpha:', alpha)
    print('sigma:', sigma)

    # Compare
    R, p = fit.distribution_compare('power_law', 'lognormal')
    print("R:", R, 'p-value', p)

    fig, ax = plt.subplots(figsize=(5, 4))

    fit.plot_pdf(color='#d62728', linewidth=2, label='Empirical data', ax=ax)

    #
    Rp = '$R = {R:.2f}$; $p = {p:.3f}$'.format(R=R, p=p)
    ax.annotate(Rp, xy=(.03, .13), xycoords='axes fraction', color='black')

    if R > 0:
        pw_goodness = '$\sigma = {sigma:.3f}$'.format(sigma=fit.power_law.sigma)
        ax.annotate(pw_goodness, xy=(.03, .05), xycoords='axes fraction', color='#1f77b4')
    else:
        ln_goodness = '$\mu = {mu:.2f}; \sigma = {sigma:.3f}$'.format(mu=fit.lognormal.mu, sigma=fit.lognormal.sigma)
        ax.annotate(ln_goodness, xy=(.03, .05), xycoords='axes fraction', color='#2ca02c')
    #
    pw_label = r'Power law fit'
    ln_label = r'Lognormal fit'
    #ex_label = r'Lognormal fit'
    fit.power_law.plot_pdf(color='#aec7e8', linewidth=1, linestyle='--', label=pw_label, ax=ax)
    fit.lognormal.plot_pdf(color='#98df8a', linewidth=1, linestyle='--', label=ln_label, ax=ax)
    #fit.exponential.plot_pdf(color='#c5b0d5', linewidth=1, linestyle='--', label=ex_label, ax=ax)

    #
    ax.set_title(r'Semi-metric edges ($s_{{ij}}>1)$' '\n' '{source:s} - {project:s} ({normalization:s})'.format(source=source_str, project=project_str, normalization=normalization))
    ax.set_ylabel(r'$P(s_{ij} \geq x)$')
    ax.set_xlabel(r'$s_{ij}$ frequency')

    ax.grid()

    ax.legend(loc='best')

    plt.tight_layout()
    # plt.subplots_adjust(left=0.09, right=0.98, bottom=0.07, top=0.90, wspace=0, hspace=0.0)
    plt.savefig(wImgFile, dpi=150, bbox_inches='tight')  # , pad_inches=0.05)


if __name__ == '__main__':
    #
    # Init
    #
    source = 'sociopatterns'  # sociopatterns, salathe, toth
    # # salathe: high-school
    # # sociopatterns: exhibit, high-school, hospital, conference, primary-school, workplace
    # # toth: elementary-school, middle-school
    project = 'exhibit'
    normalization = 'social'  # social, time, time_all
    # time_window = '20S'

    # # For Toth projects
    # # - elementary-school = ['2013-01-31','2013-02-01','all']
    # # - middle-school = :['2012-11-28','2012-11-29','all']
    date = '2009-07-15'
    #date = 'all'
    #date = None

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
    plot_s_dist(source, project, normalization, date)
