# coding=utf-8
# Author: Rion B Correia
# Date: Dec 05, 2021
#
# Description:
# Plot sankey plot of community attributions
#
# Plotting
import matplotlib as mpl
mpl.use('Agg')
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
import matplotlib.pyplot as plt
# General
from collections import defaultdict
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 40)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
# Networks
from itertools import product
import networkx as nx
from utils import *
import seaborn as sns

# ---------------
# From PySankey
class PySankeyException(Exception):
    pass


class NullsInFrame(PySankeyException):
    pass


class LabelMismatch(PySankeyException):
    pass


def check_data_matches_labels(labels, data, side):
    if len(labels > 0):
        if isinstance(data, list):
            data = set(data)
        if isinstance(data, pd.Series):
            data = set(data.unique().tolist())
        if isinstance(labels, list):
            labels = set(labels)
        if labels != data:
            msg = "\n"
            if len(labels) <= 20:
                msg = "Labels: " + ",".join(labels) + "\n"
            if len(data) < 20:
                msg += "Data: " + ",".join(data)
            raise LabelMismatch('{0} labels and data do not match.{1}'.format(side, msg))
# ---------------

# Set product comparison
def product_comparison(Alabels, Asets, Blabels, Bsets):
    r = []
    for Alabel, Blabel in product(Alabels, Blabels):
        inter = len(Asets[Alabel].intersection(Bsets[Blabel]))
        if inter > 0:
            r.append([Alabel, Blabel, inter])
    return r



def compute_sankey_data(source, project, normalization, time_window, date=None):

    # Init
    if date is not None:
        rGpickle = 'results/%s/%s/%s/graph-%s.gpickle' % (source, project, date, normalization)
    else:
        rGpickle = 'results/%s/%s/graph-%s.gpickle' % (source, project, normalization)

    if source == 'sociopatterns':
        if project == 'primary-school':
            module_attribute = 'class'
        elif project == 'high-school':
            module_attribute = 'class'
        elif project == 'hospital':
            module_attribute = 'type'
        elif project == 'workplace':
            module_attribute = 'class'
        elif project == 'exhibit':
            module_attribute = None
        elif project == 'conference':
            module_attribute = None
        else:
            ValueError("Sociopatterns project not found.")
    elif source == 'salanthe':
        if project == 'high-school':
            module_attribute = 'role'
        else:
            ValueError("Salanthe project not found.")
    elif source == 'toth':
        if project == 'elementary-school':
            module_attribute = 'grade'
        elif project == 'middle-school':
            module_attribute = 'grade'
        else:
            ValueError("Toth project not found.")
    else:
        raise ValueError("Source not found.")

    # Load Network
    print('--- Loading Network gPickle ---')
    G = nx.read_gpickle(rGpickle)

    # Stats
    e_original = 0
    e_metric = 0
    # e_ultrametric = 0
    for eid, (i, j, d) in enumerate(G.edges(data=True), start=0):
        # Original Edges
        if d.get('original') is True:
            e_original += 1
        # Metric Edges
        if d.get('metric') is True:
            e_metric += 1
        # UltraMetric Edges
        # if d.get('ultrametric') is True:
        #    e_ultrametric += 1

    # Exceptions
    if source == 'sociopatterns':
        if project == 'primary-school':
            Gp = G.copy()
            Gp.nodes[1753]['class'] = '1A'
            Gp.nodes[1745]['class'] = '1B'
            Gp.nodes[1852]['class'] = '2A'
            Gp.nodes[1650]['class'] = '2B'
            Gp.nodes[1746]['class'] = '3A'
            Gp.nodes[1709]['class'] = '3B'
            Gp.nodes[1653]['class'] = '4A'
            Gp.nodes[1521]['class'] = '4B'
            Gp.nodes[1668]['class'] = '5A'
            Gp.nodes[1824]['class'] = '5B'
        else:
            Gp = G
    else:
        Gp = G

    GOp = generate_original_graph(Gp)
    GOp = compute_louvain(GOp)

    # Meta
    meta_n, meta_s, meta_sM, _ = get_graph_variables(GOp, module_attribute)
    meta_s = sorted(meta_s)

    # Original
    orig_n, orig_s, orig_sM, _ = get_graph_variables(GOp, 'module-louvain')
    orig_s = sorted(orig_s)

    # Metric
    GMp = generate_metric_graph(GOp)
    GMp = compute_louvain(GMp)
    metr_n, metr_s, metr_sM, _ = get_graph_variables(GMp, 'module-louvain')
    metr_s = sorted(metr_s)

    # Theshold
    GTp = generate_threshold_graph(GOp, e_metric)
    GTp = compute_louvain(GTp)
    thre_n, thre_s, thre_sM, _ = get_graph_variables(GTp, 'module-louvain')
    thre_s = sorted(thre_s)

    # Random
    GRp = generate_random_graph(GOp, e_original - e_metric)
    GRp = compute_louvain(GRp)
    rand_n, rand_s, rand_sM, _ = get_graph_variables(GRp, 'module-louvain')
    rand_s = sorted(rand_s)

    # Meta vs Original
    meta_vs_orig = product_comparison(meta_s, meta_sM, orig_s, orig_sM)
    df_meta_vs_orig = pd.DataFrame(meta_vs_orig, columns=['Meta', 'Original', 'value'])
    df_meta_vs_orig['Original'] = df_meta_vs_orig['Original'].apply(lambda x: 'D' + str(x))

    # Original vs Metric
    orig_vs_metr = product_comparison(orig_s, orig_sM, metr_s, metr_sM)
    df_orig_vs_metr = pd.DataFrame(orig_vs_metr, columns=['Original', 'Metric', 'value'])
    df_orig_vs_metr['Original'] = df_orig_vs_metr['Original'].apply(lambda x: 'D' + str(x))
    df_orig_vs_metr['Metric'] = df_orig_vs_metr['Metric'].apply(lambda x: 'B' + str(x))

    # Original vs Threshold
    orig_vs_thre = product_comparison(orig_s, orig_sM, thre_s, thre_sM)
    df_orig_vs_thre = pd.DataFrame(orig_vs_thre, columns=['Original', 'Threshold', 'value'])
    df_orig_vs_thre['Original'] = df_orig_vs_thre['Original'].apply(lambda x: 'D' + str(x))
    df_orig_vs_thre['Threshold'] = df_orig_vs_thre['Threshold'].apply(lambda x: 'T' + str(x))

    # Original vs Random
    orig_vs_rand = product_comparison(orig_s, orig_sM, rand_s, rand_sM)
    df_orig_vs_rand = pd.DataFrame(orig_vs_rand, columns=['Original', 'Random', 'value'])
    df_orig_vs_rand['Original'] = df_orig_vs_rand['Original'].apply(lambda x: 'D' + str(x))
    df_orig_vs_rand['Random'] = df_orig_vs_rand['Random'].apply(lambda x: 'R' + str(x))

    return df_meta_vs_orig, df_orig_vs_metr, df_orig_vs_thre, df_orig_vs_rand


def sankey(source, project, normalization, date, df, A, B, colorDict=None, *args, **kwargs):
    # Init
    if date is not None:
        wImgFile = "images/{source:s}/{project:s}/{date:s}/{normalization:s}/sankey-{A:s}-vs-{B:s}.pdf".format(source=source, project=project, date=date, normalization=normalization, A=A, B=B)
        project_str = project.replace('-', ' ').title() + ' - ' + date.title()
    else:
        wImgFile = "images/{source:s}/{project:s}/{normalization:s}/sankey-{A:s}-vs-{B:s}.pdf".format(source=source, project=project, normalization=normalization, A=A, B=B)
        project_str = project.replace('-', ' ').title()

    if source == 'sociopatterns':
        source_str = 'SocioPatterns'
    else:
        source_str = source.title()

    if normalization == 'social':
        normalization_str = "social"
    elif normalization == 'time':
        normalization_str = "Ind. time"
    elif normalization == 'time_all':
        normalization_str = "Exp. time"

    if A == 'Meta':
        A_str = 'Metalabels'
    elif A == 'Original':
        A_str = r'Louvain on $D(X)$'

    if B == 'Original':
        B_str = r'Louvain on $D(X)$'
    elif B == 'Metric':
        B_str = r'$B(X)$'
    elif B == 'Threshold':
        B_str = 'Thresholded subgraph'
    elif B == 'Random':
        B_str = 'Random subgraph'

    title = '{A:s} vs {B:s}\n{source:s} - {project:s}'.format(A=A_str, B=B_str, source=source_str, project=project_str, normalization=normalization_str)

    left = df[A]
    right = df[B]
    leftWeight = df['value']
    rightWeight = None
    leftLabels = None
    rightLabels = None
    aspect = 20

    # Plot
    fig, ax = plt.subplots(figsize=(4, 3.5))

    # ---------------
    # Based on PySankey
    # ---------------
    if leftWeight is None:
        leftWeight = []
    if rightWeight is None:
        rightWeight = []
    if leftLabels is None:
        leftLabels = []
    if rightLabels is None:
        rightLabels = []
    rightColor = False
    showRightLabels = kwargs.get('showRightLabels', True)

    # Check weights
    if len(leftWeight) == 0:
        leftWeight = np.ones(len(left))

    if len(rightWeight) == 0:
        rightWeight = leftWeight

    # Create Dataframe
    if isinstance(left, pd.Series):
        left = left.reset_index(drop=True)
    if isinstance(right, pd.Series):
        right = right.reset_index(drop=True)
    dataFrame = pd.DataFrame({'left': left, 'right': right, 'leftWeight': leftWeight,
                              'rightWeight': rightWeight}, index=range(len(left)))
    if len(dataFrame[(dataFrame.left.isnull()) | (dataFrame.right.isnull())]):
        raise NullsInFrame('Sankey graph does not support null values.')

    # Identify all labels that appear 'left' or 'right'
    allLabels = pd.Series(np.r_[dataFrame.left.unique(), dataFrame.right.unique()]).unique()

    # Identify left labels
    if len(leftLabels) == 0:
        leftLabels = pd.Series(dataFrame.left.unique()).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame['left'], 'left')

    # Identify right labels
    if len(rightLabels) == 0:
        rightLabels = pd.Series(dataFrame.right.unique()).unique()
    else:
        check_data_matches_labels(leftLabels, dataFrame['right'], 'right')
    # If no colorDict given, make one
    if colorDict is None:
        colorDict = {}
        palette = "hls"
        colorPalette = sns.color_palette(palette, len(allLabels))
        for i, label in enumerate(allLabels):
            colorDict[label] = colorPalette[i]
    else:
        missing = [str(label) for label in allLabels if label not in colorDict.keys()]
        if missing:
            msg = "The colorDict parameter is missing values for the following labels : "
            msg += '{}'.format(', '.join(missing))
            raise ValueError(msg)

    # Determine widths of individual strips
    ns_l = defaultdict()
    ns_r = defaultdict()
    for leftLabel in leftLabels:
        leftDict = {}
        rightDict = {}
        for rightLabel in rightLabels:
            leftDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].leftWeight.sum()
            rightDict[rightLabel] = dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)].rightWeight.sum()
        ns_l[leftLabel] = leftDict
        ns_r[leftLabel] = rightDict

    # margin between items
    margin = 0.05
    margin_l = 1 / len(leftLabels)
    margin_r = 1 / len(rightLabels)

    # Determine positions of left label patches and total widths
    leftWidths = defaultdict()
    for i, leftLabel in enumerate(leftLabels):
        myD = {}
        myD['left'] = dataFrame[dataFrame.left == leftLabel].leftWeight.sum()
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['left']
        else:
            myD['bottom'] = leftWidths[leftLabels[i - 1]]['top'] + margin_l * dataFrame.leftWeight.sum()
            myD['top'] = myD['bottom'] + myD['left']
            topEdge = myD['top']
        leftWidths[leftLabel] = myD

    # Determine positions of right label patches and total widths
    rightWidths = defaultdict()
    for i, rightLabel in enumerate(rightLabels):
        myD = {}
        myD['right'] = dataFrame[dataFrame.right == rightLabel].rightWeight.sum()
        if i == 0:
            myD['bottom'] = 0
            myD['top'] = myD['right']
        else:
            myD['bottom'] = rightWidths[rightLabels[i - 1]]['top'] + margin_r * dataFrame.rightWeight.sum()
            myD['top'] = myD['bottom'] + myD['right']
            topEdge = myD['top']
        rightWidths[rightLabel] = myD

    # Total vertical extent of diagram
    xMax = topEdge / aspect

    # Draw vertical bars on left and right of each  label's section & print label
    for leftLabel in leftLabels:
        ax.fill_between(
            [-0.03 * xMax, 0],
            2 * [leftWidths[leftLabel]['bottom']],
            2 * [leftWidths[leftLabel]['bottom'] + leftWidths[leftLabel]['left']],
            color=colorDict[leftLabel],
            alpha=1.0
        )
        ax.text(
            -0.05 * xMax,
            leftWidths[leftLabel]['bottom'] + 0.5 * leftWidths[leftLabel]['left'],
            leftLabel,
            {'ha': 'right', 'va': 'center'}
        )
    for rightLabel in rightLabels:
        ax.fill_between(
            [xMax, 1.03 * xMax], 2 * [rightWidths[rightLabel]['bottom']],
            2 * [rightWidths[rightLabel]['bottom'] + rightWidths[rightLabel]['right']],
            color=colorDict[rightLabel],
            alpha=0.99
        )
        if showRightLabels:
            ax.text(
                1.05 * xMax,
                rightWidths[rightLabel]['bottom'] + 0.5 * rightWidths[rightLabel]['right'],
                rightLabel,
                {'ha': 'left', 'va': 'center'}
            )

    if not showRightLabels:
        n = dataFrame.right.unique().shape[0]
        ax.text(
            1.06 * xMax, 320,
            r'$m = {n:d}$ modules'.format(n=n),
            {'ha': 'left', 'va': 'center', 'rotation': 90}
        )

    # Plot strips
    for leftLabel in leftLabels:
        for rightLabel in rightLabels:
            labelColor = leftLabel
            if rightColor:
                labelColor = rightLabel
            if len(dataFrame[(dataFrame.left == leftLabel) & (dataFrame.right == rightLabel)]) > 0:
                # Create array of y values for each strip, half at left value,
                # half at right, convolve
                ys_d = np.array(50 * [leftWidths[leftLabel]['bottom']] + 50 * [rightWidths[rightLabel]['bottom']])
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_d = np.convolve(ys_d, 0.05 * np.ones(20), mode='valid')
                ys_u = np.array(50 * [leftWidths[leftLabel]['bottom'] + ns_l[leftLabel][rightLabel]] + 50 * [rightWidths[rightLabel]['bottom'] + ns_r[leftLabel][rightLabel]])
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')
                ys_u = np.convolve(ys_u, 0.05 * np.ones(20), mode='valid')

                # Update bottom edges at each label so next strip starts at the right place
                leftWidths[leftLabel]['bottom'] += ns_l[leftLabel][rightLabel]
                rightWidths[rightLabel]['bottom'] += ns_r[leftLabel][rightLabel]
                plt.fill_between(
                    np.linspace(0, xMax, len(ys_d)), ys_d, ys_u, alpha=0.65,
                    color=colorDict[labelColor]
                )
    ax.axis('off')
    #plt.gcf().set_size_inches(6, 6)
    ax.set_title(title)

    #plt.tight_layout()    
    plt.subplots_adjust(left=0.09, right=0.96, bottom=0.01, top=0.85, wspace=0, hspace=0.0)
    plt.savefig(wImgFile, dpi=150)
    plt.close()


if __name__ == '__main__':
    #
    # Init
    #
    source = 'sociopatterns'  # sociopatterns, salathe, toth
    # # salathe: high-school
    # # sociopatterns: exhibit, high-school, hospital, conference, primary-school, workplace
    # # toth: elementary-school, middle-school
    project = 'high-school'
    normalization = 'social'  # social, time, time_all
    time_window = '20S'

    # # For Toth projects
    # # - elementary-school = ['2013-01-31','2013-02-01','all']
    # # - middle-school = :['2012-11-28','2012-11-29','all']
    #date = 'all'
    date = None

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

    

    if source == 'sociopatterns' and project == 'high-school':

        df_meta_vs_orig, df_orig_vs_metr, df_orig_vs_thre, df_orig_vs_rand = compute_sankey_data(source, project, normalization, time_window, date)

        orig_categories = ['D8', 'D4', 'D7', 'D9', 'D2', 'D6', 'D5', 'D0', 'D1', 'D3']
        #
        # Meta vs Original (manual layout)
        #
        df = pd.DataFrame([
            ('MP*2', 'D8', 35),
            ('MP*1', 'D4', 27),
            ('MP*2', 'D7', 3),
            ('MP*1', 'D7', 2),
            ('MP', 'D4', 4),
            ('MP', 'D9', 2),
            ('MP', 'D7', 27),
            ('PSI*', 'D2', 34),
            ('PC*', 'D2', 1),
            ('PC*', 'D6', 38),
            ('PC', 'D2', 1),
            ('PC', 'D5', 41),
            ('PC', 'D6', 2),
            ('2BIO3', 'D0', 40),
            ('2BIO2', 'D0', 3),
            ('2BIO2', 'D1', 30),
            ('2BIO2', 'D3', 1),
            ('2BIO1', 'D3', 36),
        ], columns=['Meta', 'Original', 'value'])

        red1, red2, red3 = '#ff0000', '#ff6666', '#990000'
        blue1, blue2, blue3 = '#0000ff', '#00007f', '#7f7fff'
        green1, green2 = '#66b266', '#004c00'
        orange1 = '#ffa500'
        gray = 'gray'

        colorDict = {
            # 2BIO (Red)
            '2BIO1': red1, 'D3': red1,
            '2BIO2': red2, 'D1': red2,
            '2BIO3': red3, 'D0': red3,
            # MP (Blue)
            'MP': blue1, 'D7': blue1, 'D9': blue1,
            'MP*1': blue2, 'D4': blue2,
            'MP*2': blue3, 'D8': blue3,
            # PC (Green)
            'PC': green1, 'D5': green1,
            'PC*': green2, 'D6': green2,
            # PSI (Orange)
            'PSI*': orange1, 'D2': orange1,
        }
        sankey(source, project, normalization, date, df=df, A='Meta', B='Original', colorDict=colorDict)

        #
        # Original vs Metric
        #
        df = df_orig_vs_metr.copy()
        # Sort
        df['Original'] = pd.Categorical(df['Original'], categories=orig_categories, ordered=True)
        df = df.sort_values('Original').reset_index()
        # Color
        colorDict = {
            # 2BIO (Red)
            'D3': red1, 'B8': red1, 'B3': red1, 'B15': red1,
            'D1': red2, 'B1': red2, 'B8': red2,
            'D0': red3, 'B0': red3, 'B14': red3, 'B9': red3,
            # MP (Blue)
            'D7': blue1, 'D9': blue1, 'B10': blue1, 'B11': blue1, 'B13': blue1,
            'D4': blue2, 'B4': blue2, 'B5': blue2,
            'D8': blue3, 'B16': blue3, 'B12': blue3,
            # PC (Green)
            'D5': green1, 'B6': green1,
            'D6': green2, 'B7': green2,
            # PSI (Orange)
            'D2': orange1, 'B2': orange1,
        }
        sankey(source, project, normalization, date, df=df, A='Original', B='Metric', colorDict=colorDict, showRightLabels=False)

        #
        # Original vs Threshold
        #
        df = df_orig_vs_thre.copy()
        # Sort
        df['Original'] = pd.Categorical(df['Original'], categories=orig_categories, ordered=True)
        df = df.sort_values('Original').reset_index()
        # Color
        colorDict = {'D3': red1, 'D1': red2, 'D0': red3, 'D7': blue1, 'D9': blue1, 'D4': blue2, 'D8': blue3, 'D5': green1, 'D6': green2, 'D2': orange1}
        colorDict.update({label: gray for label in df['Threshold'].tolist()})
        # Plot
        sankey(source, project, normalization, date, df=df, A='Original', B='Threshold', colorDict=colorDict, showRightLabels=False)

        #
        # Original vs Random
        #
        df = df_orig_vs_rand.copy()
        # Sort
        df['Original'] = pd.Categorical(df['Original'], categories=orig_categories, ordered=True)
        df = df.sort_values('Original').reset_index()
        # Color
        colorDict = {'D3': red1, 'D1': red2, 'D0': red3, 'D7': blue1, 'D9': blue1, 'D4': blue2, 'D8': blue3, 'D5': green1, 'D6': green2, 'D2': orange1}
        colorDict.update({label: gray for label in df['Random'].tolist()})
        # Plot
        sankey(source, project, normalization, date, df=df, A='Original', B='Random', colorDict=colorDict, showRightLabels=False)
