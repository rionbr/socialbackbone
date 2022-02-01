# coding=utf-8
# Author: Rion B Correia
# Date: Dec 05, 2017
# 
# Description: 
# Plot Neighbour distributions from Networks
#
from __future__ import division
#Plotting
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
# General
from collections import OrderedDict
import pandas as pd
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
# Networks
import networkx as nx
import utils


def _calculate_lambda(G, attribute):
    n = G.number_of_nodes()
    ks = dict()
    for k in range(1, n):
        print('computing {:d} of {:d} nodes'.format(k, n))
        vdict = []
        for i in G.nodes():
            # iclass = G.nodes[i][attribute]
            # Select only neighbor edges
            edges = [(i, j, G.edges[i, j]['distance']) for j in G.neighbors(i)]
            # Sort by proximity and cut list by k
            kedges = sorted(edges, key=lambda x: x[2])[:k]
            # Compute lambda
            lmbd = sum([1 if G.nodes[i][attribute] == G.nodes[j][attribute] else 0 for _, j, p in kedges]) / k
            vdict.append(lmbd)
        L = sum(vdict) / n
        ks[k] = L
    return ks


def calculate_lambda(source, project, normalization, time_window, module_attribute='class', date=None):
    print('--- Ploting: %s : %s : %s : %s : %s ---' % (source, project, normalization, time_window, date))

    # Init
    if date is not None:
        rGpickle = 'results/%s/%s/%s/graph-%s.gpickle' % (source, project, date, normalization)
    else:
        rGpickle = 'results/%s/%s/graph-%s.gpickle' % (source, project, normalization)

    # Load Network
    print('> Loading Network gPickle')
    G = nx.read_gpickle(rGpickle)

    if date is not None:
        wCSVFile = "results/%s/%s/%s/neighbours-%s.csv" % (
            source, project, date, normalization
        )
    else:
        wCSVFile = "results/%s/%s/neighbours-%s.csv" % (
            source, project, normalization
        )

    # Exceptions
    print('--- Computing Neighbour ---')
    e_original = sum([1 for i, j, d in G.edges(data=True) if 'original' in d])
    e_metric = sum([1 for i, j, d in G.edges(data=True) if d.get('metric') == True])

    print('> ks Original')
    GO = utils.generate_original_graph(G)
    ks_o = _calculate_lambda(GO, module_attribute)

    print('> ks Metric ')
    #GO = utils.generate_original_graph(G)
    GM = utils.generate_metric_graph(G)
    ks_m = _calculate_lambda(GM, module_attribute)

    print('> ks Threshold')
    #GO = utils.generate_original_graph(G)
    GT = utils.generate_threshold_graph(GO, e_metric)
    ks_t = _calculate_lambda(GT, module_attribute)

    print('> ks Random')
    #GO = utils.generate_original_graph(G)
    GR = utils.generate_random_graph(GO, e_original - e_metric)
    ks_r = _calculate_lambda(GR, module_attribute)

    print('> Building csv')
    df = pd.DataFrame(
        OrderedDict([
            ('edges', ks_o.keys()),
            ('ks-original', ks_o.values()),
            ('ks-metric', ks_m.values()),
            ('ks-threshold', ks_t.values()),
            ('ks-random', ks_r.values()),
        ])
    )
    print(df)

    print('> Saving csv')
    df.to_csv(wCSVFile, encoding='utf-8')


if __name__ == '__main__':
    #
    # Init
    #
    source = 'toth'  # sociopatterns, salanthe, toth
    ## salanthe: high-school
    ## sociopatterns: high-school, hospital, primary-school, workplace
    ## toth: elementary-school, middle-school
    project = 'middle-school' 
    normalization = 'social'  # social, time, time_all
    time_window = '20S'
    module_attribute = 'grade'  # class, type, role, grade

    ## For Toth projects
    ## - elementary-school = ['2013-01-31','2013-02-01','all']
    ## - middle-school = :['2012-11-28','2012-11-29','all']
    date = 'all'
    #date = None

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

            calculate_lambda(source, project, normalization, time_window, module_attribute, date=datestr)

    else:
        calculate_lambda(source, project, normalization, time_window, module_attribute, date=date)
