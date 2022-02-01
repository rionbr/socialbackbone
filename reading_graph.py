# coding=utf-8
# Author: Rion B Correia
# Date: Jan 16, 2018
#
# Description: Tutorial
# Read the .gpickle file and extracts only the original and backbone of the networks.
#
import networkx as nx
import gzip
import utils
import pandas as pd


def extract_subgraphs(source, project, normalization, time_window, date=None):

    # Init
    if date is not None:
        rGpickle = 'results/%s/%s/%s/graph-%s.gpickle' % (source, project, date, normalization)
        # Original Network Files
        wOgraphml = 'results/%s/%s/%s/original/graph-%s-original.graphml' % (source, project, date, normalization)
        wOpickle = 'results/%s/%s/%s/original/graph-%s-original.gpickle' % (source, project, date, normalization)
        wOcsv = 'results/%s/%s/%s/original/graph-%s-original.csv.gz' % (source, project, date, normalization)
        # Metric Backbone Files
        wBgraphml = 'results/%s/%s/%s/metric/graph-%s-metric.graphml' % (source, project, date, normalization)
        wBpickle = 'results/%s/%s/%s/metric/graph-%s-metric.gpickle' % (source, project, date, normalization)
        wBcsv = 'results/%s/%s/%s/metric/graph-%s-metric.csv.gz' % (source, project, date, normalization)
        #
        # original_name = '%s - %s - %s - %s - %s Original' % (source, project, normalization, time_window, date)
        backbone_name = '%s - %s - %s - %s - %s Metric Backbone' % (source, project, normalization, time_window, date)

    else:
        rGpickle = 'results/%s/%s/graph-%s.gpickle' % (source, project, normalization)
        # Original Network Files
        wOgraphml = 'results/%s/%s/original/graph-%s-original.graphml' % (source, project, normalization)
        wOpickle = 'results/%s/%s/original/graph-%s-original.gpickle' % (source, project, normalization)
        wOcsv = 'results/%s/%s/original/graph-%s-original.csv.gz' % (source, project, normalization)
        # Metric Backbone Files
        wBgraphml = 'results/%s/%s/metric/graph-%s-metric.graphml' % (source, project, normalization)
        wBpickle = 'results/%s/%s/metric/graph-%s-metric.gpickle' % (source, project, normalization)
        wBcsv = 'results/%s/%s/metric/graph-%s-metric.csv.gz' % (source, project, normalization)
        #
        # original_name = '%s - %s - %s - %s Original' % (source, project, normalization, time_window)
        backbone_name = '%s - %s - %s - %s Metric Backbone' % (source, project, normalization, time_window)

    # Read Graph
    G = nx.read_gpickle(rGpickle)

    #
    # Extract edges in metric backbone
    #
    print('-- Backbone Graph --')
    backbone_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('metric') == True]

    # Builds new Backbone Graph
    B = nx.Graph(backbone_edges, name=backbone_name)

    # Export backbone to other formats
    print('> Graphml')
    utils.ensurePathExists(wBgraphml)
    nx.write_graphml(B, wBgraphml)
    print('> gPickle')
    utils.ensurePathExists(wBpickle)
    nx.write_gpickle(B, wBpickle)
    print('> Adjacency Matrix (List)')
    utils.ensurePathExists(wBcsv)
    dfG = nx.to_pandas_edgelist(B)
    zfile = gzip.open(wBcsv, 'wb')
    zfile.write(dfG.to_csv(encoding='utf-8'))
    zfile.close()

    #
    # Extract edges in the original graph
    #
    print('-- Original Graph --')
    original_edges = [(u, v, d) for u, v, d in G.edges(data=True) if d.get('original') == True]

    # Builds new Backbone Graph
    B2 = nx.Graph(original_edges, name=backbone_name)

    # Export backbone to other formats
    print('> Graphml')
    utils.ensurePathExists(wOgraphml)
    nx.write_graphml(B2, wOgraphml)
    print('> gPickle')
    utils.ensurePathExists(wOpickle)
    nx.write_gpickle(B2, wOpickle)
    print('> Adjacency Matrix (List)')
    utils.ensurePathExists(wOcsv)
    dfG = nx.to_pandas_edgelist(B2)
    zfile = gzip.open(wOcsv, 'wb')
    zfile.write(dfG.to_csv(encoding='utf-8'))
    zfile.close()


if __name__ == '__main__':
    #
    # Init
    #
    source = 'sociopatterns'  # sociopatterns, salanthe, toth
    # # salanthe: high-school
    # # sociopatterns: exhibit, high-school, hospital, conference, primary-school, workplace
    # # toth: elementary-school, middle-school
    project = 'exhibit'
    normalization = 'social'  # social, time, time_all
    time_window = '20S'

    # # For Toth projects
    # # - elementary-school = ['2013-01-31','2013-02-01','all']
    # # - middle-school = :['2012-11-28','2012-11-29','all']
    # date = None
    # date = 'all'

    if project == 'exhibit':
        # start='2009-04-28', end='2009-07-17'
        date_range = pd.date_range(start='2009-07-15', end='2009-07-16')
        for date in date_range:
            datestr = date.strftime('%Y-%m-%d')

            # Some dates do not exist, just skip
            if datestr in ['2009-05-04', '2009-05-08', '2009-05-11', '2009-05-18', '2009-05-25', '2009-06-01',
                           '2009-06-08', '2009-06-15', '2009-06-22', '2009-06-29', '2009-07-06', '2009-07-13']:
                continue
            extract_subgraphs(source, project, normalization, time_window, date=datestr)

    else:
        extract_subgraphs(source, project, normalization, time_window, date=date)
