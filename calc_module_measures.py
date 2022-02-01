# coding=utf-8
# Author: Rion B Correia
# Date: Dec 05, 2017
#
# Description:
# Calculates Module Measures
#
from collections import OrderedDict
import pandas as pd
import math
from scipy.stats import entropy
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
pd.set_option('precision', 2)
# Networks
import networkx as nx
from utils import *
# Matplotlib
import matplotlib as mpl
mpl.use('Agg')
# clusim
from clusim.clustering import Clustering
from clusim.clusimelement import element_sim  # , element_sim_elscore
# Adjusted Rand Index
from sklearn.metrics import adjusted_rand_score


def tuple2columns(r):
    if isinstance(r, float):
        return pd.Series([r, None, None])
    elif isinstance(r, tuple):
        return pd.Series([None, r[0], r[1]])
    else:
        raise TypeError('Something is wrong. Should be either a float or a tuple')


def calc_h(A, B):
    """
        Calculates h_{A \to B}. See paper for details.

        Returns:
            ( h_{A \to B} , h_{B \to A} ) : float
    """
    # Intersection
    df_I = pd.DataFrame.from_dict(OrderedDict([(i, OrderedDict([(j, iv.intersection(jv)) for j, jv in B.items()])) for i, iv in A.items()]), orient='index')
    df_Ic = df_I.applymap(len)  # size of sets
    ma, mb = df_I.shape
    # to == 'B'
    sA = pd.Series(A, name='A')
    sAc = sA.apply(len)
    df_BA = df_Ic.divide(sAc.values, axis='index')
    sH = df_BA.apply(entropy, axis=1, base=2)
    h_A2B = (1 / (ma * math.log(mb, 2))) * sH.sum()
    # to =='A'
    sB = pd.Series(B, name='B')
    sBc = sB.apply(len)
    df_AB = df_Ic.divide(sBc.values, axis='columns')
    sH = df_AB.apply(entropy, axis=0, base=2)
    h_B2A = (1 / (mb * math.log(ma, 2))) * sH.sum()
    #
    return h_A2B, h_B2A


def calc_y(A, B):
    """
        Calculates y_{A, B}. See paper for details.

        Returns:
            y_{A, B} : float
    """
    df_I = pd.DataFrame.from_dict(OrderedDict([(i, OrderedDict([(j, iv.intersection(jv)) for j, jv in B.items()])) for i, iv in A.items()]), orient='index')
    df_U = pd.DataFrame.from_dict(OrderedDict([(i, OrderedDict([(j, iv.union(jv)) for j, jv in B.items()])) for i, iv in A.items()]), orient='index')
    ma, mb = df_I.shape
    df_Ic = df_I.applymap(len)
    df_Uc = df_U.applymap(len)
    df_S = df_Ic.divide(df_Uc)
    return df_S.sum().sum() / math.sqrt(ma * mb)


def calc_J(A, B):
    """
        Calculates J_{A \to B}. See paper for details.

        Returns:
            ( j_{A \to B} , j_{B \to A} ) : float
    """
    df_I = pd.DataFrame.from_dict(OrderedDict([(i, OrderedDict([(j, iv.intersection(jv)) for j, jv in B.items()])) for i, iv in A.items()]), orient='index')
    df_U = pd.DataFrame.from_dict(OrderedDict([(i, OrderedDict([(j, iv.union(jv)) for j, jv in B.items()])) for i, iv in A.items()]), orient='index')
    ma, mb = df_I.shape
    df_Ic = df_I.applymap(len)
    df_Uc = df_U.applymap(len)
    df_S = df_Ic.divide(df_Uc)
    sMaxA = df_S.max(axis=0)
    sMaxB = df_S.max(axis=1)
    jA2B = sMaxB.sum() / ma
    jB2A = sMaxA.sum() / mb
    return jA2B, jB2A


def calc_clusim(A, B):
    cA = Clustering(clus2elm_dict=A)
    cB = Clustering(clus2elm_dict=B)
    return element_sim(cA, cB)


def calc_ari(A, B):
    """ Adjusted Rand Index"""
    A = {v: k for k, s in A.items() for v in s}
    B = {v: k for k, s in B.items() for v in s}
    df = pd.DataFrame({'A': A, 'B': B}).dropna()
    A = df['A'].to_list()
    B = df['B'].to_list()
    return adjusted_rand_score(A, B)


def calculate_modules_measures(source, project, normalization, time_window, module_attribute='class', date=None):
    print('--- Calculating: %s : %s : %s : %s : %s ---' % (source, project, normalization, time_window, date))

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
    if source == 'sociopatterns':

        if project == 'primary-school':
            print('> Applying node change exceptions (teachers to same class as students)')
            _G = G.copy()
            G.nodes[1753]['class'] = '1A'
            G.nodes[1745]['class'] = '1B'
            G.nodes[1852]['class'] = '2A'
            G.nodes[1650]['class'] = '2B'
            G.nodes[1746]['class'] = '3A'
            G.nodes[1709]['class'] = '3B'
            G.nodes[1653]['class'] = '4A'
            G.nodes[1521]['class'] = '4B'
            G.nodes[1668]['class'] = '5A'
            G.nodes[1824]['class'] = '5B'

        if project == 'workplace':
            print('> Applying exception (removing isolates)')
            _G = G.copy()
            G.remove_nodes_from(list(nx.isolates(G)))

    elif source == 'toth':
        if project == 'middle-school':
            print('> Removing unknown class nodes')
            # removing nodes [69, 111, 140, 246, 355]
            nodes2remove = [i for i, d in G.nodes(data=True) if d.get(module_attribute, None) == 'Unknown']
            G.remove_nodes_from(nodes2remove)

    e_original = sum([1 for i, j, d in G.edges(data=True) if 'original' in d])
    e_metric = sum([1 for i, j, d in G.edges(data=True) if d.get('metric') == True])
    e_ultrametric = sum([1 for i, j, d in G.edges(data=True) if d.get('ultrametric') == True])

    #
    #
    #
    print('--- Computing Measures ---')
    n = list()
    d = list()

    #
    #
    #
    if '_G' in locals():
        _GO = generate_original_graph(_G)
        _GO = compute_louvain(_GO)
        _nO, _, _mO, _ = get_graph_variables(_GO, module_attribute)
        _nOl, _, mOl, _ = get_graph_variables(_GO, 'module-louvain')
    else:
        _nO = '--'
        _nOl = '--'
    #
    GO = generate_original_graph(G)
    GM = generate_metric_graph(GO)
    GU = generate_ultrametric_graph(G)
    GT = generate_threshold_graph(GO, e_metric)
    GO = compute_louvain(GO)  # ; compute_infomap(GO)
    GM = compute_louvain(GM)  # ; compute_infomap(GM)
    GU = compute_louvain(GU)  # ; compute_infomap(GU)
    GT = compute_louvain(GT)  # ; compute_infomap(GT)

    nO, _, mO, _ = get_graph_variables(GO, module_attribute)
    nOl, _, mOl, _ = get_graph_variables(GO, 'module-louvain')
    # nOi, _, mOi, _ = get_graph_variables(GO, 'module-infomap')
    #
    nMl, _, mMl, _ = get_graph_variables(GM, 'module-louvain')
    # nMi, _, mMi, _ = get_graph_variables(GM, 'module-infomap')

    nTl, _, mTl, _ = get_graph_variables(GT, 'module-louvain')
    # nTi, _, mTi, _ = get_graph_variables(GT, 'module-infomap')

    # Random
    li = []
    li.extend([
        ('Original', 'Metalabels', _nO),
        ('Original', 'Louvain', _nOl),
        ('Original(*)', 'Metalabels', nO),
        ('Original(*)', 'Louvain', nOl),
        ('Metric', 'Louvain', nMl),
        ('Threshold', 'Louvain', nTl),
        # ('Random', 'Louvain', nRl), # COMES FROM RANDOM
        # ('Metric', 'Infomap', nMi),
        # ('Threshold', 'Infomap', nTi),
        # ('Random', 'Infomap', nRi), # COMES FROM RANDOM

    ])
    dfM = pd.DataFrame(li, columns=['method', 'measure', 'value'])
    print('> dfM (number of clusters)')
    print(dfM)
    print('* Exception case. When labels are changed (eg., primary-school)')

    # Metalabels -> Other (If there are MetaLabels)
    if module_attribute is not None:
        d.extend([
            ('Metalabels', 'Proximity', 'Louvain', 'y', calc_y(mO, mOl)),
            ('Metalabels', 'Proximity', 'Louvain', 'J', calc_J(mO, mOl)),
            ('Metalabels', 'Proximity', 'Louvain', 'h', calc_h(mO, mOl)),
            ('Metalabels', 'Proximity', 'Louvain', 'clusim', calc_clusim(mO, mOl)),
            ('Metalabels', 'Proximity', 'Louvain', 'ari', calc_ari(mO, mOl)),
            # ('Metalabels','Proximity','Infomap','h', calc_h(mO,mOi)),
            # ('Metalabels','Proximity','Infomap','y', calc_y(mO,mOi)),
            # ('Metalabels','Proximity','Infomap','J', calc_J(mO,mOi)),
            #
            ('Metalabels', 'Metric', 'Louvain', 'y', calc_y(mO, mMl)),
            ('Metalabels', 'Metric', 'Louvain', 'J', calc_J(mO, mMl)),
            ('Metalabels', 'Metric', 'Louvain', 'h', calc_h(mO, mMl)),
            ('Metalabels', 'Metric', 'Louvain', 'clusim', calc_clusim(mO, mMl)),
            ('Metalabels', 'Metric', 'Louvain', 'ari', calc_ari(mO, mMl)),
            # ('Metalabels', 'Metric', 'Infomap', 'y', calc_y(mO,mMi)),
            # ('Metalabels', 'Metric', 'Infomap', 'J', calc_J(mO,mMi)),
            # ('Metalabels', 'Metric', 'Infomap', 'h', calc_h(mO,mMi)),
            #
            # ('Metalabels', 'Ultrametric', 'Louvain', 'y', calc_y(mO,mUl)),
            # ('Metalabels', 'Ultrametric', 'Louvain', 'J', calc_J(mO,mUl)),
            # ('Metalabels', 'Ultrametric', 'Louvain', 'h', calc_h(mO,mUl)),
            # ('Metalabels', 'Ultrametric', 'Infomap', 'y', calc_y(mO,mUi)),
            # ('Metalabels', 'Ultrametric', 'Infomap', 'J', calc_J(mO,mUi)),
            # ('Metalabels', 'Ultrametric', 'Infomap', 'h', calc_h(mO,mUi)),

            ('Metalabels', 'Threshold', 'Louvain', 'h', calc_h(mO, mTl)),
            ('Metalabels', 'Threshold', 'Louvain', 'J', calc_J(mO, mTl)),
            ('Metalabels', 'Threshold', 'Louvain', 'y', calc_y(mO, mTl)),
            ('Metalabels', 'Threshold', 'Louvain', 'clusim', calc_clusim(mO, mTl)),
            ('Metalabels', 'Threshold', 'Louvain', 'ari', calc_ari(mO, mTl)),
            # ('Metalabels', 'Threshold', 'Infomap', 'y', calc_y(mO,mTi)),
            # ('Metalabels', 'Threshold', 'Infomap', 'J', calc_J(mO,mTi)),
            # ('Metalabels', 'Threshold', 'Infomap', 'h', calc_h(mO,mTi)),
        ])
    # Proximity -> Others
    d.extend([
        ('Proximity', 'Metric', 'Louvain', 'y', calc_y(mOl, mMl)),
        ('Proximity', 'Metric', 'Louvain', 'J', calc_J(mOl, mMl)),
        ('Proximity', 'Metric', 'Louvain', 'h', calc_h(mOl, mMl)),
        ('Proximity', 'Metric', 'Louvain', 'clusim', calc_clusim(mOl, mMl)),
        ('Proximity', 'Metric', 'Louvain', 'ari', calc_ari(mOl, mMl)),
        # ('Proximity', 'Metric', 'Infomap', 'y', calc_y(mOi, mMi)),
        # ('Proximity', 'Metric', 'Infomap', 'J', calc_J(mOi, mMi)),
        # ('Proximity', 'Metric', 'Infomap', 'h', calc_h(mOi, mMi)),
        #
        # ('Proximity', 'Ultrametric', 'Louvain', 'y', calc_y(mOl, mUl)),
        # ('Proximity', 'Ultrametric', 'Louvain', 'J', calc_J(mOl, mUl)),
        # ('Proximity', 'Ultrametric', 'Louvain', 'h', calc_h(mOl, mUl)),
        # ('Proximity', 'Ultrametric', 'Infomap', 'y', calc_y(mOi, mUi)),
        # ('Proximity', 'Ultrametric', 'Infomap', 'J', calc_J(mOi, mUi)),
        # ('Proximity', 'Ultrametric', 'Infomap', 'h', calc_h(mOi, mUi)),

        ('Proximity', 'Threshold', 'Louvain', 'y', calc_y(mOl, mTl)),
        ('Proximity', 'Threshold', 'Louvain', 'J', calc_J(mOl, mTl)),
        ('Proximity', 'Threshold', 'Louvain', 'h', calc_h(mOl, mTl)),
        ('Proximity', 'Threshold', 'Louvain', 'clusim', calc_clusim(mOl, mTl)),
        ('Proximity', 'Threshold', 'Louvain', 'ari', calc_ari(mOl, mTl)),
        # ('Proximity', 'Threshold', 'Infomap', 'h', calc_h(mOi, mTi)),
        # ('Proximity', 'Threshold', 'Infomap', 'J', calc_J(mOi, mTi)),
        # ('Proximity', 'Threshold', 'Infomap', 'y', calc_y(mOi, mTi)),
    ])

    df = pd.DataFrame(d, columns=['A', 'B', 'method', 'measure', 'value'])
    df['measure'] = df['measure'].astype(pd.api.types.CategoricalDtype(categories=['y', 'J', 'h', 'clusim', 'ari'], ordered=True))
    df.set_index(['A', 'B', 'method', 'measure'], inplace=True)
    df[['AB', 'A->B', 'A<-B']] = df['value'].apply(tuple2columns)
    df.drop('value', axis='columns', inplace=True)

    print('> df (measures)')
    for idx, dft in df.groupby(level=3, sort=False):
        print('Measure: {:s}'.format(idx))
        print(dft.dropna(axis='columns'))
        print('---')

    #
    # Random
    #
    print('**************')
    print('--- Random ---')
    print('**************')
    lr = []
    r = []
    for GR in generate_n_random_graphs(GO, n=100, edges_to_keep=e_original - e_metric):
        GR = compute_louvain(GR)  # ; compute_infomap(GR)
        nR, _, mR, _ = get_graph_variables(GR, module_attribute)
        nRl, _, mRl, _ = get_graph_variables(GR, 'module-louvain')
        # nRi, _, mRi, _ = get_graph_variables(GR, 'module-infomap')
        lr.extend([
            ('Random', 'Louvain', nRl)
            # ('Random', 'Infomap', nRi)
        ])
        if module_attribute is not None:
            r.extend([
                ('Metalabels', 'Random', 'Louvain', 'h', calc_h(mO, mRl)),
                ('Metalabels', 'Random', 'Louvain', 'y', calc_y(mO, mRl)),
                ('Metalabels', 'Random', 'Louvain', 'J', calc_J(mO, mRl)),
                ('Metalabels', 'Random', 'Louvain', 'clusim', calc_clusim(mO, mRl)),
                ('Metalabels', 'Random', 'Louvain', 'ari', calc_ari(mO, mRl)),
                # ('Metalabels', 'Random', 'Infomap', 'h', calc_h(mO, mRi)),
                # ('Metalabels', 'Random', 'Infomap', 'y', calc_y(mO, mRi)),
                # ('Metalabels', 'Random', 'Infomap', 'h', calc_J(mO, mRi)),
            ])
        r.extend([
            ('Proximity', 'Random', 'Louvain', 'h', calc_h(mOl, mRl)),
            ('Proximity', 'Random', 'Louvain', 'y', calc_y(mOl, mRl)),
            ('Proximity', 'Random', 'Louvain', 'J', calc_J(mOl, mRl)),
            ('Proximity', 'Random', 'Louvain', 'clusim', calc_clusim(mOl, mRl)),
            ('Proximity', 'Random', 'Louvain', 'ari', calc_ari(mOl, mRl)),
            # ('Proximity', 'Random', 'Infomap', 'h', calc_h(mO, mRi)),
            # ('Proximity', 'Random', 'Infomap', 'y', calc_y(mO, mRi)),
            # ('Proximity', 'Random', 'Infomap', 'h', calc_J(mO, mRi)),
        ])
    dfMr = pd.DataFrame(lr, columns=['method', 'measure', 'value'])
    dfMr = dfMr.groupby(['method', 'measure']).agg(['mean', 'std'])
    print('> dfMr (number of clusters)')
    print(dfMr)

    dfR = pd.DataFrame(r, columns=['A', 'B', 'method', 'measure', 'value'])
    dfR['measure'] = dfR['measure'].astype(pd.api.types.CategoricalDtype(categories=['y', 'J', 'h', 'clusim', 'ari'], ordered=True))
    dfR.set_index(['A', 'B', 'method', 'measure'], inplace=True)
    dfR[['AB', 'A->B', 'A<-B']] = dfR['value'].apply(tuple2columns)
    dfR.drop('value', axis='columns', inplace=True)
    dfRg = dfR.groupby(level=[0, 1, 2, 3]).agg(['mean', 'std'])

    print('> dfR (measures)')
    for idx, dft in dfRg.groupby(level=3, sort=False):
        print('Measure: {:s}'.format(idx))
        print(dft.dropna(axis='columns'))
        print('--- ---')


if __name__ == '__main__':
    #
    # Init
    #
    source = 'toth'  # sociopatterns, salathe, toth
    # # salanthe: high-school (role)
    # # sociopatterns: conference (None), high-school (class), hospital (type), primary-school (class), workplace (class)
    # # toth: elementary-school (class/grade), middle-school (grade)
    project = 'middle-school'
    normalization = 'social'  # social, time, time_all
    time_window = '20S'
    module_attribute = 'grade'  # class, type, role, grade, None

    # # For Toth projects
    # # - elementary-school = ['2013-01-31','2013-02-01','all']
    # # - middle-school = :['2012-11-28','2012-11-29','all']
    date = 'all'
    #date = None

    # Social, Individual Time, Experiment Time

    if project == 'exhibit':
        # start='2009-04-28', end='2009-07-17'
        date = '2009-07-15'
        calculate_modules_measures(source, project, normalization, time_window, module_attribute, date=date)
        """
        date_range = pd.date_range(start='2009-07-15', end='2009-07-15')
        for date in date_range:
            datestr = date.strftime('%Y-%m-%d')

            # Some dates do not exist, just skip
            if datestr in ['2009-05-04','2009-05-08','2009-05-11','2009-05-18','2009-05-25','2009-06-01',
                '2009-06-08','2009-06-15','2009-06-22','2009-06-29','2009-07-06','2009-07-13']:
                continue

            calculate_modules_measures(source, project, normalization, time_window, module_attribute, date=datestr)
        """
    else:
        calculate_modules_measures(source, project, normalization, time_window, module_attribute, date=date)
