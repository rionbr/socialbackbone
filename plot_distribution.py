# coding=utf-8
# Author: Rion B Correia
# Date: Dec 05, 2017
# 
# Description: 
# Plot Distributions of Edge Values from Networks
#
from __future__ import division
#Plotting
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt
# General
import pandas as pd
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
# Networks
import networkx as nx
import utils


def plot_distributions(source, project, normalization, time_window, plottypes, date=None):

    print('--- Ploting: %s : %s : %s : %s : %s ---' % (source, project, normalization, time_window, date))

    # Init
    if date is not None:
        rGpickle = 'results/%s/%s/%s/graph-%s.gpickle' % (source, project, date, normalization)
        project_str = project.replace('-', ' ').title() + ' ' + date

    else:
        rGpickle = 'results/%s/%s/graph-%s.gpickle' % (source, project, normalization)
        project_str = project.replace('-', ' ').title()

    source_str = source.title()

    # Load Network
    print('> Loading Network gPickle')
    G = nx.read_gpickle(rGpickle)

    # Exceptions
    if source == 'sociopatterns':
        if project == 'workplace':
            print('> Applying exception (removing isolates)')
            _G = G.copy()
            G.remove_nodes_from(list(nx.isolates(G)))

    if normalization == 'social':
        normalization_str = "social"
    elif normalization == 'time':
        normalization_str = "ind. time"
    elif normalization == 'time_all':
        normalization_str = "exp. time"

    for dp in plottypes:
        plottype = dp['plottype']
        loglog = dp['loglog']

        print("Plotting %s" % (plottype))

        if date is not None:
            wPNGFile = "images/%s/%s/%s/%s/distrib-%s%s.pdf" % (
                source, project, date, normalization, plottype, ('-ll' if loglog else '')
            )
        else:
            wPNGFile = "images/%s/%s/%s/distrib-%s%s.pdf" % (
                source, project, normalization, plottype, ('-ll' if loglog else '')
            )
        #
        # SubGraphs
        #

        # Original Graph, without the metric closure
        G_original = G.copy()
        edges2remove = [(i, j) for i, j, d in G.edges(data=True) if 'original' in d]
        G_original.remove_edges_from(edges2remove)

        if plottype == 'degree':

            title = u'Node degree distribution\n%s-%s-%s' % (source_str, project_str, normalization_str)
            values = [d for n, d in G_original.degree()]
            ylabel = r'$degree(i)$'
            xlabel = r'$rank$'
            color = 'orange'
            edgecolor = 'darkorange'

        elif plottype == 'proximity':

            title = u'Edge proximity distribution\n%s - %s (%s)' % (source_str, project_str, normalization_str)
            values = [d['weight'] for i, j, d in G.edges(data=True) if 'original' in d]
            ylabel = r'$p_{ij}$'
            xlabel = r'$rank$'
            color = 'none'
            edgecolor = '#9467bd'  # purple

        elif plottype == 'distance':

            title = u'Edge distance distribution\n%s - %s (%s)' % (source_str, project_str, normalization_str)
            values = [d['distance'] for i, j, d in G.edges(data=True) if 'original' in d]
            ylabel = r'$d_{ij}$'
            xlabel = r'$rank$'
            color = 'none'
            edgecolor = '#1f77b4'  # blue

        elif plottype == 'metric':

            title = u'Metric-backbone edge proximity distribution\n%s-%s-%s' % (source_str, project_str, normalization_str)
            values = [d['distance'] for i, j, d in G.edges(data=True) if d['metric'] == True]
            ylabel = r'$d_{ij}$'
            xlabel = r'$rank$'
            color = 'darkred'
            edgecolor = 'maroon'

        elif plottype == 'semi-metric':

            title = u'Semi-metric edge proximity distribution\n%s-%s-%s' % (source_str, project_str, normalization_str)
            values = [d['distance'] for i, j, d in G.edges(data=True) if d['metric'] == False]
            ylabel = r'$p_{ij}$'
            xlabel = r'$rank$'
            color = 'darkblue'
            edgecolor = 'midnightblue'

        elif plottype == 's-value':

            title = u'S values distribution\n%s-%s-%s' % (source_str, project_str, normalization_str)
            values = [d['s_value'] for i, j, d in G.edges(data=True) if 's_value' in d]
            ylabel = r'$s_{ij}$'
            xlabel = r'$rank$'
            color = 'goldenrod'
            edgecolor = 'darkgoldenrod'

        elif plottype == 'b-value':

            title = u'B values distribution\n%s-%s-%s' % (source_str, project_str, normalization_str)
            values = [d['b_ij_value'] for i, j, d in G.edges(data=True) if 'b_ij_value' in d]
            ylabel = r'$b_{ij}$'
            xlabel = r'$rank$'
            color = 'magenta'
            edgecolor = 'purple'
        else:
            raise ValueError('PlotType value not found.')

        sV = pd.Series(values, name=plottype)
        sV = sV.sort_values(ascending=False).reset_index(drop=True)

        if sV.shape[0] > 1000:
            print('- Rasterizing points')
            rasterized = True
        else:
            rasterized = False
        #
        fig, (ax) = plt.subplots(figsize=(4, 3), nrows=1, ncols=1)
        plt.rc('font', size=10)
        # plt.rc('legend', fontsize=12)
        plt.rc('legend', numpoints=1)
        plt.rc('axes', titlesize=11)
        plt.rc('axes', labelsize=14)

        # ax.axis('equal')

        # Plot

        ax.plot(sV.index, sV.values,
                markerfacecolor=color, markeredgecolor=edgecolor,
                alpha=1.0, linestyle='', marker='o', markersize=6,
                rasterized=rasterized
                )

        ax.set_title(title)
        ax.set_ylabel(ylabel, fontsize='large')
        ax.set_xlabel(xlabel, fontsize='large')

        if (loglog):
            # ax.set_xscale("log")
            ax.set_yscale("log")

        ax.grid()

        utils.ensurePathExists(wPNGFile)
        # plt.tight_layout()
        plt.savefig(wPNGFile, dpi=150, bbox_inches='tight', pad_inches=0.05)
        plt.close()


if __name__ == '__main__':
    #
    # Init
    #
    source = 'sociopatterns'  # sociopatterns, salanthe, toth
    # # salanthe: high-school
    # # sociopatterns: exhibit, high-school, hospital, conference, primary-school, workplace
    # # toth: elementary-school, middle-school
    project = 'hospital'
    normalization = 'time_all'  # social, time, time_all
    time_window = '20S'

    # # For Toth projects
    # # - elementary-school = ['2013-01-31','2013-02-01','all']
    # # - middle-school = :['2012-11-28','2012-11-29','all']
    # date = 'all'
    date = None

    # Social, Individual Time, Experiment Time

    plottypes = [
        #{'plottype': 'degree', 'loglog': False},
        {'plottype': 'proximity', 'loglog': True},
        #{'plottype': 'distance', 'loglog': True},
        # {'plottype': 's-value'  , 'loglog': True},
        # {'plottype': 'b-value'  , 'loglog': True},
        #
        # {'plottype': 'proximity', 'loglog': False},
        # {'plottype': 'distance' , 'loglog': False},
    ]

    if project == 'exhibit':
        # start='2009-04-28', end='2009-07-17'
        date_range = pd.date_range(start='2009-07-15', end='2009-07-15')
        for date in date_range:
            datestr = date.strftime('%Y-%m-%d')

            # Some dates do not exist, just skip
            if datestr in ['2009-05-04', '2009-05-08', '2009-05-11', '2009-05-18', '2009-05-25', '2009-06-01',
                           '2009-06-08', '2009-06-15', '2009-06-22', '2009-06-29', '2009-07-06', '2009-07-13']:
                continue
            plot_distributions(source, project, normalization, time_window, plottypes, date=datestr)

    else:
        plot_distributions(source, project, normalization, time_window, plottypes, date=date)
