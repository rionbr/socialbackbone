import os
# import sys
import random
import networkx as nx
import community  # louvain
import infomap


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


def generate_random_graph(G, edges_to_keep=0):
    GR = G.copy()
    edges2remove = random.sample(list(GR.edges(data=True)), edges_to_keep)
    GR.remove_edges_from(edges2remove)
    return GR


def generate_n_random_graphs(G, n=1, *arg, **kwargs):
    for i in range(n):
        yield generate_random_graph(G, *arg, **kwargs)


def generate_threshold_graph(G, edges_to_keep=0):
    GT = G.copy()
    edges2remove = sorted(GT.edges(data=True), key=lambda x: x[2]['weight'])[:-edges_to_keep]
    GT.remove_edges_from(edges2remove)
    return GT


def get_graph_variables(G, *arg, **kwargs):
    dM = nx.get_node_attributes(G, *arg)
    s = set(dM.values())
    n = len(s)
    sM = {m: set([k for k, v in dM.items() if v == m]) for m in s}
    #
    return n, s, sM, dM


def compute_louvain(G):
    G = G.copy()
    dM = community.best_partition(G, random_state=123)
    nx.set_node_attributes(G, name='module-louvain', values=dM)
    return G


def compute_infomap(G):
    G = G.copy()
    infomapWrapper = infomap.Infomap("--two-level --undirected --silent")
    # Building Infomap network from a NetworkX graph...
    dto = {n: i for i, n in enumerate(G.nodes(), start=0)}

    dfrom = {i: n for n, i in dto.items()}
    for i, j, d in G.edges(data=True):
        i = dto[i]
        j = dto[j]
        infomapWrapper.addLink(i, j, d['proximity'])
    # Run!
    infomapWrapper.run()
    tree = infomapWrapper.tree
    # Dict of Results
    dM = {}
    for node in tree.leafIter():
        i = dfrom[node.originalLeafIndex]
        dM[i] = node.moduleIndex()
    # s = set(dM.values())

    nx.set_node_attributes(G, name='module-infomap', values=dM)
    return G


def ensurePathExists(filepath):
    """Given a file with path, ensures a path exists by creating a directory if needed. """
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
