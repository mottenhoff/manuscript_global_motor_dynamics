from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from matplotlib.patches import Rectangle
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_kullback_sym
from pyriemann.utils.covariance import normalize
from pyriemann.stats import PermutationDistance
from scipy.stats import mannwhitneyu

COVMATS, LABELS = 0, 1

def get_sig_level(p, n_tests=1):

    p *= n_tests # Bonferonni

    if p < 0.001: return '***'
    if p < 0.01:  return '**'
    if p < 0.05:  return '*'

    return 'n.s.'


def get_data(path):
    dirs = [d for d in path.iterdir() if d.is_dir() and 'pc' not in d.stem]
    dirs = sorted(dirs, key=lambda x:int(x.stem[2:]))

    data =   [np.load(d/f'processed_data.npy') for d in sorted(dirs)]
    labels = [np.load(d/f'labels.npy')         for d in sorted(dirs)]
    return data, labels

def get_covs(x):
    x = x.transpose(0, 2, 1)
    return Covariances(estimator='lwf').transform(x)

def get_covs_per_class(data):

    for k, v in data.items():
        covs = [get_covs(d) for d in v[COVMATS]]
        data[k] = list(data[k])
        data[k][COVMATS] = covs 

    move_trials, rest_trials = [], []

    for cov, label in zip(data[k][COVMATS], data[k][LABELS]):
        is_move = np.where(np.isin(label, ['right', 'left']), True, False)

        move_trials.append(cov[is_move])
        rest_trials.append(cov[~is_move])

    move, rest = np.vstack(move_trials), np.vstack(rest_trials)
    move_labels = np.full(move.shape[0], 'move')
    rest_labels = np.full(rest.shape[0], 'rest')

    return move, move_labels, rest, rest_labels

def plot_panel(ax, move, rest):

    color = cm.plasma(0)

    move, rest = move[:, 0, 0], rest[:, 0, 0]

    _, p = mannwhitneyu(rest, move, alternative='two-sided')

    # plot
    vp_rest = ax.violinplot(rest, positions=[0], showmeans=False, showmedians=True, showextrema=False)
    vp_move = ax.violinplot(move, positions=[1], showmeans=False, showmedians=True, showextrema=False)

    for vp in [vp_rest, vp_move]:
        vp['cmedians'].set_color('black')
        vp['bodies'][0].set_facecolor(color)
        vp['bodies'][0].set_edgecolor(color)
        vp['bodies'][0].set_alpha(.5)

    xlim = ax.get_xlim()
    ax.add_patch(Rectangle((xlim[0], 0), np.abs(xlim[0])-0.01, max(rest), color='white'))
    ax.add_patch(Rectangle((.5, 0), 0.49, max(move), color='white'))

    ax.set_yscale('log')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Rest', 'Move'], fontsize='x-large')

    ax.set_xlim(-0.325, 1.325)

    # Significance line
    height = np.max([rest.max(), move.max()]) * 1.1
    offset = height*0.1
    xmin, xmax = 0, 1

    is_sig = get_sig_level(p, n_tests=1)

    xlim = ax.get_xlim()
    l = np.abs(xlim).sum()
    xmin_frac = (xmin + np.abs(xlim[0]))/l
    xmax_frac = (xmax + np.abs(xlim[0]))/l

    ax.axhline(height, xmin_frac, xmax_frac, color='k', linewidth=2)
    ax.annotate(f'{is_sig} (p={p:.3f})', xy=((xmax+xmin)/2, height), xytext=((xmax+xmin)/2, height+offset),
                ha='center', va='bottom', fontsize='xx-large')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_ylabel('Variance in PC 1', fontsize='xx-large')

    return ax


def make(path):

    tasks =   ['grasp']
    filters = ['beta']

    fig, ax = plt.subplots()

    move, rest = np.load('move.npy'), np.load('rest.npy')
    ax = plot_panel(ax, move, rest)

    fig.savefig('./figures/figure_8.png')
    fig.savefig('./figures/figure_8.svg')

    return ax
