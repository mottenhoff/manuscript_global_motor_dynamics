from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats

from libs import mappings

FONTSIZE = 25

PCS = 0
PPTS = 1
FOLDS = 2

AUC = 0
BAC = 1
TRAIN = 0
TEST = 1

def fdrcorrection(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    # p = np.asfarray(p)
    p = np.array(p, dtype=np.float64)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def annotate_min_max(ax):

    line = ax.lines[0]
    x_values, y_values = line.get_xdata(), line.get_ydata()
    x, y = x_values[np.argmax(y_values)], np.max(y_values)

    ax.annotate(np.round(y, 2), (x, y), xytext=(x-1, y+0.05), fontsize=FONTSIZE/2)

    return ax

def get_data(path, type_):
    # data = [pcs, ppts, folds, metrics, train/test]

    dirs = [d for d in path.iterdir() if d.is_dir() and 'pc' in d.stem]
    dirs = sorted(dirs, key=lambda x:int(x.stem[2:]))

    data = []
    for d in dirs:
        data += [np.stack([np.load(ppt/f'{type_}.npy') for ppt in d.iterdir()])]

    return np.stack(data)

def plot_panel(ax, data):

    pcs = np.concatenate([[3], np.arange(5, 51, 5)])

    ppts = np.arange(data.shape[1])
    colors = [mappings.color_map()[f'p{i+1}'] for i in ppts]

    aucs = data[:, :, :, AUC, TEST].reshape(data.shape[0], np.multiply(*data.shape[1:3]))

    # STD calculation here is different than the manuscript. Here is it the std over all folds,
    #     in the paper its the std over the means, resulting in a smaller std in the paper
    # See this line: np.std([np.mean(aucs[:, s:e]) for s, e in zip(np.arange(0, 72, 8), np.arange(8, 80, 8))])
    mean, std = aucs.mean(axis=1), aucs.std(axis=1)

    t, p = stats.ttest_1samp(aucs, .5, axis=1)
    p = fdrcorrection(p)

    facecolors = np.where(p<0.05, 'black', 'lightgrey')
    edgecolors = ['black'] * facecolors.size

    ax.plot(pcs, mean, color='k', zorder=1)
    ax.scatter(pcs, mean, facecolors=facecolors, edgecolors=edgecolors, s=25, zorder=2)
    ax.fill_between(pcs, mean-std, mean+std, alpha=0.15, color='k')

    ax = annotate_min_max(ax)

    for i, ppt in enumerate(np.arange(data.shape[PPTS])):
        ax.plot(pcs, data[:, ppt, :, AUC, TEST].mean(axis=1), color=colors[i], alpha=0.3)

    ax.axhline(0.5, linestyle='--', color='black', alpha=0.7)
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks(pcs)

    return ax

def make(path):
    filters = ('beta', 'hg', 'betahg')

    data = [get_data(path/filter_, 'full') for filter_ in filters]

    N_COLS = 3
    N_ROWS = 1


    ax_idc = np.arange(N_COLS)

    fig, axs = plt.subplots(nrows=N_ROWS, ncols=N_COLS, figsize=(16, 9))
    fig.suptitle('')

    for idx, d in zip(ax_idc, data):
        plot_panel(axs[idx], d)

    fig.savefig(r'./figures/figure_4_cross_task.png')
    fig.savefig(r'./figures/figure_4_cross_task.svg')

    return fig, axs
