# coding=utf-8
# Author: Rion B Correia & Nathan Ratkiewicz
# Date: Dec 08, 2017
#
# Description:
# Print Statistics of Edges from Networks
#
from __future__ import division
# General
from collections import OrderedDict
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 40)
pd.set_option('display.width', 1000)
pd.set_option('precision', 4)
# Networks
import community # install 'python-louvain' first
import networkx as nx
# Distance Closure
from distanceclosure.utils import dist2prox


def generate_original_graph(G):
    GO = G.copy()
    edges2remove = [(i, j) for i, j, d in G.edges(data=True) if 'original' not in d]
    GO.remove_edges_from(edges2remove)
    return GO


def generate_metric_graph(G):
    GM = G.copy()
    edges2remove = [(i, j) for i, j, d in G.edges(data=True) if d['metric'] == False]
    GM.remove_edges_from(edges2remove)
    return GM


def generate_ultrametric_graph(G):
    GU = G.copy()
    edges2remove = [(i, j) for i, j, d in G.edges(data=True) if d['ultrametric'] == False]
    GU.remove_edges_from(edges2remove)
    return GU


def print_statistics(source, project, normalization, time_window, date=None):
    # Init
    if date is not None:
        rGpickle = 'results/%s/%s/%s/graph-%s.gpickle' % (source, project, date, normalization)
    else:
        rGpickle = 'results/%s/%s/graph-%s.gpickle' % (source, project, normalization)

    # Load Network
    print('--- Loading Network gPickle (%s) ---' % (rGpickle))
    G = nx.read_gpickle(rGpickle)
    print('Network: %s' % (G.name))
    #
    # SubGraphs
    #
    # Original Graph, without the metric closure
    GO = generate_original_graph(G)
    GM = generate_metric_graph(GO)
    GU = generate_ultrametric_graph(GO)

    """
    print('> Clustering (C)')
    CO = nx.average_clustering(GO)
    CM = nx.average_clustering(GM)
    CU = nx.average_clustering(GU)

    print('> Community detection (Q) Louvain')
    best_partition_original = community.best_partition(GO)
    QO = community.modularity(best_partition_original, GO)
    best_partition_metric = community.best_partition(GM)
    QU = community.modularity(best_partition_metric, GM)
    best_partition_ultrametric = community.best_partition(GU)
    QU = community.modularity(best_partition_ultrametric, GU)
    """
    #
    # Statistics
    #
    print('--- Calculating Statistics ---')
    n = G.number_of_nodes()
    isolates = len(list(nx.isolates(G)))
    isolates_percent = isolates / n

    e_possible = int(((n * n) - n) / 2)
    e_total = G.number_of_edges()
    original_components = nx.number_connected_components(GO)

    e_original = 0
    e_metric = 0
    e_ultrametric = 0
    e_semimetric = 0
    e_s_gt_1 = 0
    e_s_gt_1_original = 0
    e_d_eq_infty = 0
    e_bij_gt_1 = 0
    e_bji_gt_1 = 0
    distortion = 0

    for eid, (i, j, d) in enumerate(G.edges(data=True), start=0):
        # Original Edges
        if d.get('original'):
            e_original += 1
        # Metric Edges
        if d.get('metric') == True:
            e_metric += 1
        # UltraMetric Edges
        if d.get('ultrametric') == True:
            e_ultrametric += 1

        # Semi-metric edges
        if (d.get('metric') == False):
            e_semimetric += 1

        # S values
        if d.get('s_value') is not None:
            #
            if d.get('s_value') > 1.0:
                e_s_gt_1 += 1
            #
            if d.get('s_value') > 1.0 and d.get('original') == True:
                e_s_gt_1_original += 1

        if d.get('distance') == np.inf:
            e_d_eq_infty += 1

        # B_ij and B_ji values
        if (d.get('b_ij_value') is not None) or (d.get('b_ji_value') is not None):
            # B_ij values
            if (d.get('b_ij_value')) > 1.0:
                e_bij_gt_1 += 1
            # B_ji values
            if (d.get('b_ji_value')) > 1.0:
                e_bji_gt_1 += 1

        # Distortion
        distortion += abs(dist2prox(d['distance_metric_closure']) - d.get('proximity'))

    distortion_norm = (2 * distortion) / (n * (n - 1))
    #e_original_percent = e_original / e_total
    e_metric_percent = e_metric / e_original
    e_ultrametric_percent = e_ultrametric / e_original
    #e_semimetric_percent = e_semimetric/e_total
    #e_s_gt_1_percent = e_s_gt_1 / e_original
    e_s_gt_1_original_percent = e_s_gt_1_original / e_original
    #e_d_eq_infty_percent = e_d_eq_infty / e_original
    e_bij_gt_1_percent = e_bij_gt_1 / e_semimetric
    e_bji_gt_1_percent = e_bji_gt_1 / e_semimetric

    print()
    print('-- D_w -- ')
    print('Nodes:             {:,d}'.format(n))
    # print('Possible Edges:    {:,d}'.format(e_possible))
    print('Edges:             {:,d}'.format(e_original))
    # print('Conn. Components:  {:,d}'.format(original_components))
    if original_components > 1:
        print('Isolates:          {:,d} ({:.2%})'.format(isolates, isolates_percent))
    if (original_components == 1) and (e_possible != e_total):
        raise Exception('After Closure, the graph must be fully connected ({:d} != {:d}), given there was only one connected component'.format(e_possible, e_total))
    # print('Avg. Clustering (C): {:,d}'.format(CO))
    # print('Modularity:          {:,d}'.format(QO))
    print()
    print('-- B_w --')
    print('Metric Edges:      {:,d} ({:.2%} of original)'.format(e_metric, e_metric_percent))
    print('UltraMetric Edges: {:,d} ({:.2%} of original)'.format(e_ultrametric, e_ultrametric_percent))
    # print('Avg. Clustering (C): {:,d}'.format(CM))
    # print('Modularity:          {:,d}'.format(QM))

    print()
    print('-- D^C_w --')
    print('Semi-Metric edges: {:,d}'.format(e_semimetric))
    print('S>1:               {:,d} ({:.2%})'.format(e_s_gt_1, e_s_gt_1_original_percent))
    print('B_ij>1:            {:,d} ({:.2%})'.format(e_bij_gt_1, e_bij_gt_1_percent))
    print('B_ji>1:            {:,d} ({:.2%})'.format(e_bji_gt_1, e_bji_gt_1_percent))

    print('Distortion:        {:.4f} ({:.2f})'.format(distortion_norm, distortion))
    print()

    return OrderedDict([
        ('date', date),
        ('nodes', n),
        ('edges', e_original),
        ('isolates', isolates),
        ('isolates %', isolates_percent),
        #
        ('metric', e_metric),
        ('metric %', e_metric_percent),
        ('ultra', e_ultrametric),
        ('ultra %', e_ultrametric_percent),
        #
        ('semi', e_semimetric),
        ('S>1', e_s_gt_1),
        ('S>1 %', e_s_gt_1_original_percent),
        ('B_ij>1', e_bij_gt_1),
        ('B_ij>1 %', e_bij_gt_1_percent),
        ('B_ji>1', e_bji_gt_1),
        ('B_ji>1 %', e_bji_gt_1_percent),
        #
        ('distortion_norm', distortion_norm),
        ('distortion', distortion)
    ])


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
    time_window = '20S'

    # # For Toth projects
    # # - elementary-school = ['2013-01-31','2013-02-01','all']
    # # - middle-school = :['2012-11-28','2012-11-29','all']
    # date = 'all'
    date = None

    ##
    if project == 'exhibit':

        date_range = pd.date_range(start='2009-04-28', end='2009-07-17')  # start='2009-04-28', end='2009-07-17'
        results = []
        for date in date_range:
            datestr = date.strftime('%Y-%m-%d')

            # Some dates do not exist, just skip
            if datestr in ['2009-05-04', '2009-05-08', '2009-05-11', '2009-05-18', '2009-05-25', '2009-06-01',
                           '2009-06-08', '2009-06-15', '2009-06-22', '2009-06-29', '2009-07-06', '2009-07-13']:
                continue

            result = print_statistics(source, project, normalization, time_window, date=datestr)

            results.append(result)
        dfR = pd.DataFrame(results)
        print(dfR)
        dfRg = dfR.aggregate(['mean', 'std'])
        cols0d = ['nodes', 'edges', 'isolates', 'metric', 'ultra', 'semi', 'S>1', 'B_ij>1', 'B_ji>1', 'distortion']
        cols2d = ['isolates %', 'metric %', 'ultra %', 'S>1 %', 'B_ij>1 %', 'B_ji>1 %']
        dfRg[cols0d] = dfRg[cols0d].round(decimals=0)
        dfRg[cols2d] = dfRg[cols2d].round(decimals=4)
        print(dfRg)

    else:
        print_statistics(source, project, normalization, time_window, date=date)
