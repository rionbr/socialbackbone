# coding=utf-8
# Author: Rion B Correia
# Date: Dec 08, 2017
#
# Description:
# Builds the different networks in the SocioPatterns project
#
from __future__ import division
#
# General
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
# Network
import networkx as nx
# Distance Closure
from distanceclosure.cython._dijkstra import _cy_single_source_shortest_distances
from distanceclosure._dijkstra import _py_single_source_shortest_distances
from distanceclosure.dijkstra import Dijkstra
from distanceclosure.utils import _prox2dist as prox2dist
#
from datasets import salanthe, sociopatterns, toth
import gzip
import utils


def build_network(source, project, normalization, time_window, date=None):

    #
    # Files
    #
    if date is not None:
        wGgraphml = 'results/%s/%s/%s/graph-%s.graphml' % (source, project, date, normalization)
        wGpickle = 'results/%s/%s/%s/graph-%s.gpickle' % (source, project, date, normalization)
        wGcsv = 'results/%s/%s/%s/graph-%s.csv.gz' % (source, project, date, normalization)
        network_name = '%s - %s - %s - %s - %s' % (source, project, normalization, time_window, date)
    else:
        wGgraphml = 'results/%s/%s/graph-%s.graphml' % (source, project, normalization)
        wGpickle = 'results/%s/%s/graph-%s.gpickle' % (source, project, normalization)
        wGcsv = 'results/%s/%s/graph-%s.csv.gz' % (source, project, normalization)
        network_name = '%s - %s - %s - %s' % (source, project, normalization, time_window)
    #
    # Load data
    #
    if source == 'salanthe':
        if project == 'high-school':
            dataset = salanthe.high_school()
        else:
            raise ValueError("Salanthé project not found.")

    elif source == 'sociopatterns':
        if project == 'exhibit':
            dataset = sociopatterns.exhibit(date=date)
        elif project == 'high-school':
            dataset = sociopatterns.high_school()
        elif project == 'hospital':
            dataset = sociopatterns.hospital()
        elif project == 'conference':
            dataset = sociopatterns.conference()
        elif project == 'primary-school':
            dataset = sociopatterns.primary_school()
        elif project == 'workplace':
            dataset = sociopatterns.workplace(size='large')
        else:
            raise ValueError("SocioPatterns project not found.")

    elif source == 'toth':
        if project == 'elementary-school':
            dataset = toth.elementary_school(date=date)
        elif project == 'middle-school':
            dataset = toth.middle_school(date=date)
        else:
            raise ValueError("Toth project project not found.")

    else:
        raise ValueError("Variable 'source' not found. Must be either 'sociopatterns' or 'salanthe'")

    dfC, dfM = dataset.get_contact_sequence(), dataset.get_metadata()

    print('Computing Network: %s - %s - %s - %s - %s' % (source, project, normalization, time_window, date))

    print(dfC.head())
    print(dfM.head())

    if source == 'salanthe':
        if normalization == 'social':
            dfC.reset_index(drop=True)
            dfS_ij = dfC.groupby(['i']).agg({'count': 'sum'})
            dfS_ji = dfC.groupby(['j']).agg({'count': 'sum'})
            # Change name from 'j' to 'i'
            dfS_ji.index.name = 'i'
            # Then concatenate the grouping
            dfS_i = pd.concat([dfS_ij, dfS_ji], axis=0, ignore_index=False, sort=True).groupby('i').agg('sum').astype(int).rename(columns={'count': 'social'})
            # Dict of social counts
            counts = dfS_i['social'].to_dict()

            dfX = dfC.groupby(['i', 'j']).agg({'count': 'sum'}).reset_index()
        else:
            raise ValueError("Normalization for 'Salanthé' must be 'social'.")

    elif source in ['sociopatterns', 'toth']:

        idx_min, idx_max = dfC['created_time_fmt'].min(), dfC['created_time_fmt'].max()
        T = pd.date_range(start=idx_min, end=idx_max, freq=time_window)
        # Time Grouper
        dfG = dfC.set_index('created_time_fmt').groupby(pd.Grouper(freq=time_window))

        # i,j Grouper within time window
        def f(x):
            if len(x):
                return x.groupby(['i', 'j']).agg(lambda x: True).rename(columns={'created_time': 'bool'})
            else:
                return None
        dfW = dfG.apply(f)

        dfX = pd.pivot_table(dfW, values='bool', index=['i', 'j'], aggfunc='count').reset_index(drop=False).rename(columns={'bool': 'count'})

        # Social
        if normalization == 'social':

            dfS_ij = dfW.groupby(['i']).agg('sum')
            dfS_ji = dfW.groupby(['j']).agg('sum')
            # Change name from 'j' to 'i'
            dfS_ji.index.name = 'i'
            # Then concatenate the grouping
            dfS_i = pd.concat([dfS_ij, dfS_ji], axis=0, ignore_index=False).groupby('i').agg('sum').astype(int).rename(columns={'bool': 'social'})
            # Dict of social counts
            counts = dfS_i['social'].to_dict()

        # Time
        elif normalization == 'time':

            # Group by (window,i), then by (window,j)
            dfW_ij = dfW.groupby(['created_time_fmt', 'i']).agg('count')  # .unstack(level=1)
            dfW_ji = dfW.groupby(['created_time_fmt', 'j']).agg('count')  # .unstack(level=1)
            # Then concatenate the grouping
            dfW_i = pd.concat([dfW_ij, dfW_ji], axis=0, ignore_index=False).groupby(['created_time_fmt', 'i']).agg('sum').astype(bool)
            # Dict of time counts
            counts = dfW_i.groupby('i').agg('sum')['bool'].to_dict()

        # Time All
        elif normalization == 'time_all':

            # Get the number of time windows
            num_time_windows = len(T)

            # Get unique i and j values
            unique_ij = pd.unique(dfC[['i', 'j']].values.ravel('K'))
            dfW_i = pd.DataFrame(num_time_windows, index=unique_ij, columns=['total_time'])
            counts = dfW_i.to_dict()
            counts = counts['total_time']
        else:
            raise ValueError("Normalization for 'SocioPatterns' must be either 'time', 'time_all' or 'social'.")

    else:
        raise ValueError("Source must be wither 'salanthe', 'sociopatterns', or 'toth'")

    #
    # build network
    #
    print('--- Building Network ---')
    G = nx.Graph(name=network_name)

    # Add Nodes (including metadata)
    print('> Adding Nodes')
    nodes_dict = dfM.to_dict(orient='index')
    G.add_nodes_from([(k, v) for k, v in nodes_dict.items()])

    # Add Edges
    # Note: don't have to worry about i-j and j-i, the original has i<j, always.
    print('> Adding Edges')
    edges_dict = dfX.to_dict(orient='index')
    G.add_edges_from([(v['i'], v['j'], {'count': v['count'], 'original': True}) for k, v in edges_dict.items()])

    print('Components: %d' % (nx.number_connected_components(G)))

    print('--- Compute Normalized p_ij ---')
    def normalize(i, j, d):
        r_ij = d['count']
        r_ii = counts[i]
        r_jj = counts[j]
        if (r_ii + r_jj - r_ij) >= 0:
            return r_ij / (r_ii + r_jj - r_ij)
        else:
            return 0.

    P = [ normalize(i,j,d) for i,j,d in G.edges(data=True) ]

    P_dict = dict(zip(G.edges(), P))
    D_dict = dict(zip(G.edges(), map(prox2dist, P)))

    nx.set_edge_attributes(G, name='weight', values=P_dict)
    nx.set_edge_attributes(G, name='proximity', values=P_dict)
    nx.set_edge_attributes(G, name='distance', values=D_dict)

    print('--- Computing Dijkstra APSP ---')
    dij = Dijkstra.from_edgelist(D_dict, directed=False, verbose=10)

    print('> Metric')
    poolresults = range(len(dij.N))
    for node in dij.N:
        print('> Dijkstra node %s of %s' % (node, len(dij.N)))
        poolresults[node] = _py_single_source_shortest_distances(node, dij.N, dij.E, dij.neighbours, (min, sum), verbose=2)
    shortest_distances, local_paths = map(list, zip(*poolresults))
    dij.shortest_distances = dict(zip(dij.N, shortest_distances))
    MSD = dij.get_shortest_distances(format='dict', translate=True)
    print('> Metric, done.')

    print('> Ultrametric')
    poolresults = range(len(dij.N))
    for node in dij.N:
        print('> Dijkstra node %s of %s' % (node, len(dij.N)))
        poolresults[node] = _py_single_source_shortest_distances(node, dij.N, dij.E, dij.neighbours, (min, max), verbose=2)
    shortest_distances, local_paths = map(list, zip(*poolresults))
    dij.shortest_distances = dict(zip(dij.N, shortest_distances))
    UMSD = dij.get_shortest_distances(format='dict', translate=True)
    print('> UltraMetric, done.')

    print('> Populating (G)raph : Metric')
    Cm = {(i, j): v for i, jv in MSD.iteritems() for j, v in jv.iteritems()}  # Convert Dict-of-Dicts to Dict

    # Cm contains two edges of each. Make sure we are only inserting one
    edges_seen = set()
    for (i, j), cm in Cm.iteritems():
        # Closure Network is undirected. Small ids come first
        if (i, j) in edges_seen or (i == j):
            continue
        else:
            edges_seen.add((i, j))

            # New Edge?
            if not G.has_edge(i,j):
                # Self-loops have proximity 1, non-existent have 0
                proximity = 1.0 if i == j else 0.0
                G.add_edge(i, j, distance=np.inf, proximity=proximity, distance_metric_closure=float(cm), metric=False)
            else:
                G[i][j]['distance_metric_closure'] = cm
                G[i][j]['metric'] = True if ((cm == G[i][j]['distance']) and (cm!=np.inf)) else False

    print('> Populating (G)raph : UltraMetric')
    Cum = {(i, j): v for i, jv in UMSD.iteritems() for j, v in jv.iteritems()}  # Convert Dict-of-Dicts to Dict

    # Cm contains two edges of each. Make sure we are only inserting one
    edges_seen = set()
    for (i, j), cum in Cum.iteritems():
        # Closure Network is undirected. Small ids come first
        if (i, j) in edges_seen or (i == j):
            continue
        else:
            edges_seen.add((i, j))

            # New Edge?
            if not G.has_edge(i, j):
                # Self-loops have proximity 1, non-existent have 0
                proximity = 1.0 if i == j else 0.0
                G.add_edge(i,j, distance=np.inf, proximity=proximity, distance_ultrametric_closure=float(cum), ultrametric=False)
            else:
                G[i][j]['distance_ultrametric_closure'] = cum
                G[i][j]['ultrametric'] = True if ((cum == G[i][j]['distance']) and (cum!=np.inf)) else False

    # Are there any isolates?
    isolates = list(nx.isolates(G))

    print('--- Calculating S Values ---')

    S = {
        (i, j): float(d['distance'] / d['distance_metric_closure'])
        for i, j, d in G.edges(data=True)
        if ((d.get('distance') < np.inf) and (d.get('distance_metric_closure') > 0))
    }
    nx.set_edge_attributes(G, name='s_value', values=S)

    print('--- Calculating B Values ---')
    mean_distance = {
        k: np.mean([d['distance'] for i, j, d in G.edges(nbunch=k, data=True) if 'count' in d])
        for k in G.nodes() if k not in isolates
    }
    print('> b_ij')
    B_ij = {
        (i, j): float(mean_distance[i] / d['distance_metric_closure'])
        for i, j, d in G.edges(data=True)
        if (d.get('distance') == np.inf)
    }
    nx.set_edge_attributes(G, name='b_ij_value', values=B_ij)

    print('> b_ji')
    B_ji = {
        (i, j): float(mean_distance[j] / d['distance_metric_closure'])
        for i, j, d in G.edges(data=True)
        if (d.get('distance') == np.inf)
    }
    nx.set_edge_attributes(G, name='b_ji_value', values=B_ji)

    print '--- Exporting Formats ---'
    utils.ensurePathExists(wGgraphml)
    utils.ensurePathExists(wGpickle)
    utils.ensurePathExists(wGcsv)

    print '> Graphml'
    nx.write_graphml(G, wGgraphml)
    print '> gPickle'
    nx.write_gpickle(G, wGpickle)
    print '> Adjacency Matrix (List)'
    dfG = nx.to_pandas_edgelist(G)
    zfile = gzip.open(wGcsv, 'wb')
    zfile.write(dfG.to_csv(encoding='utf-8'))
    zfile.close()


if __name__ == '__main__':
    #
    # Init
    #
    source = 'sociopatterns' # sociopatterns, salanthe, toth
    ## salanthe: high-school
    ## sociopatterns: exhibit, high-school, hospital, conference, primary-school, workplace
    ## toth: elementary-school, middle-school
    project = 'exhibit'
    normalization = 'time_all'  # social, time, time_all
    time_window = '20S'

    ## For Toth projects
    ## - elementary-school = ['2013-01-31','2013-02-01', 'all']
    ## - middle-school = :['2012-11-28','2012-11-29', 'all']
    date = 'all'
    #date = None

    if project == 'exhibit':
        # start='2009-04-28', end='2009-07-17'
        date_range = pd.date_range(start='2009-04-28', end='2009-07-17')
        for date in date_range:
            datestr = date.strftime('%Y-%m-%d')

            # Some dates do not exist, just skip
            if datestr in ['2009-05-04', '2009-05-08', '2009-05-11', '2009-05-18', '2009-05-25', '2009-06-01',
                           '2009-06-08', '2009-06-15', '2009-06-22', '2009-06-29', '2009-07-06', '2009-07-13']:
                continue

            build_network(source, project, normalization, time_window, date=datestr)

    else:
        build_network(source, project, normalization, time_window, date=date)
