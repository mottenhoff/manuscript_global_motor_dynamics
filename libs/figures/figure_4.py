from pathlib import Path
from collections import defaultdict
from itertools import permutations, product
from math import floor

import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_1samp, ttest_ind, rankdata

from libs import mappings
from libs.figures.figure_4_cross_task import plot_panel as plot_cross_task_panel
from libs.figures.figure_4_variation_in_pc import plot_panel as plot_first_components_distributions
from libs.figures.figure_4_variation_in_pc import get_covs_per_class
from libs.figures.figure_4_variation_in_pc import get_data as get_data_first_components

BETA, HG, BHG = 0, 1, 2

def fdrcorrection(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    # p = np.asfarray(p)
    p = np.array(p, dtype=np.float64)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def annote_significance_above_max_y_value(ax, y_values: list, p_values: list):

    x_ticks = ax.get_xticks()

    is_sig =  lambda p_value: '***' if p_value < 0.001 else \
                              '**'  if p_value < 0.01  else \
                              '*'   if p_value < 0.05  else \
                              'n.s'

    for x_tick, y_value, p_value in zip(x_ticks, y_values, p_values):
        ax.annotate(f'{is_sig(p_value)}', xy=(x_tick, y_value+0.05), ha='center', va='bottom', fontsize='x-large')

    return ax

def get_data(path, type_):

    # data = [pcs, ppts, folds, metrics, train/test]    

    dirs = [d for d in path.iterdir() if d.is_dir() and 'pc' in d.stem]
    dirs = sorted(dirs, key=lambda x:int(x.stem[2:]))

    data = []
    for d in dirs:
        data += [np.stack([np.load(ppt/f'{type_}.npy') for ppt in d.iterdir()])]

    return np.stack(data)

def plot_panel_source(ax, data):

    # data = 8 x 7 --> each row = source, each column is target
    data = np.vstack([np.insert(row, i, 0) for i, row in enumerate(data)])
    
    # Source
    if target := False:
        data = data.T

    data = [row[np.where(row!=0)] for row in data]

    pvalues = ttest_1samp(data, .5, axis=1, alternative='two-sided').pvalue
    pvalues = fdrcorrection(pvalues)

    ppts = sorted(mappings.kh_to_ppt().values())
    colors = [mappings.color_map()[ppt] for ppt in ppts]

    # Create a violinplot for each column of the matrix
    vp = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=False)

    for i, body in enumerate(vp['bodies']):
        body.set_facecolor('grey')

    vp['cmeans'].set_color('black')    
    
    # Customize the plot
    ax.set_xticks(np.arange(1, 9))
    ax.set_xticklabels([ppt[1:] for ppt in ppts], fontsize='x-large')

    ax.set_title('Cross-participant', fontsize='xx-large')
    ax.set_xlabel('Source participant' if not target else 'Target participant', fontsize='xx-large')

    ax.set_ylim(0.4, 1)
    ax.axhline(0.5, linestyle='--', color='black', alpha=0.7)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.tick_params(axis='both', which='major', labelsize='x-large')

    annote_significance_above_max_y_value(ax, np.array([row.max() for row in data]), pvalues)
    
    # Add individual data points using jitter
    for i, d in enumerate(data, start=1):
        x = i + np.linspace(-d.size/2, d.size/2, d.size) * 0
        cs = [tuple(c) for c in np.delete(colors, i-1, axis=0)]
        ax.scatter(x, d, s=20, c=cs, alpha=0.75)

    return ax

def add_to_cross_task_plot(ax, data, cross_data):
    AUC, TEST = 0, 1
    pcs = np.concatenate([[3], np.arange(5, 51, 5)])

    cross_data = cross_data[:, :, :, AUC, TEST].squeeze()  # shape = (11, 8, 1, 2, 2)

    aucs = data[:, :, :, AUC, TEST].reshape(data.shape[0], np.multiply(*data.shape[1:3]))

    pvals = np.array([ttest_ind(auc_c, auc_w).pvalue for auc_c, auc_w in zip(cross_data, aucs)])

    facecolors = np.where(pvals * len(pvals) < 0.05, 'black', 'lightgrey')

    ax.plot((-100, -100), (-101, -101), linestyle='solid', color='black', label='cross task')  # Hack for legend label
    ax.plot(pcs, aucs.mean(axis=1), linestyle='dashed', color='black', label='within task', zorder=1)
    ax.scatter(pcs, aucs.mean(axis=1), facecolor=facecolors, edgecolor='black', s=25, zorder=2)

    ax.tick_params(axis='both', which='major', labelsize='x-large')

    ax.legend(frameon=False, loc='upper left')

    ax.set_xlabel('Principal components', fontsize='xx-large')
    ax.set_ylim(0.4, 1)

    return ax

def plot_matrix(ax, data, within_results):

    # Add the diagonal
    # data = np.vstack([np.insert(row, i, 0) for i, row in enumerate(data)])
    data = np.vstack([np.insert(row, i, within_results[i]) for i, row in enumerate(data)])

    # mean_cross_vs_within = np.array([(np.hstack([row[:i], row[i+1:]]).mean(), row[i]) for i, row in enumerate(data.T)])  # ppt as target

    ax.imshow(data, cmap='plasma', vmin=0.25, vmax=1)

    for s, t in product(np.arange(data.shape[0]), np.arange(data.shape[1])):
        value = data[t, s]
        txt = f'{value:0.2f}' if value != 0 else 'n/a'
        ax.annotate(txt, (s, t), ha='center', va='center', color='w')

    ax.set_xlabel('Target participant', fontsize='xx-large')
    ax.set_ylabel('Source participant', fontsize='xx-large')
    
    ppts = sorted(mappings.kh_to_ppt().values())
    ppts = [ppt[1:] for ppt in ppts]

    ax.set_xticks(np.arange(len(ppts)))
    ax.set_xticklabels(ppts, fontsize='large')
    ax.set_yticks(np.arange(len(ppts)))
    ax.set_yticklabels(ppts, fontsize='large')
    ax.invert_yaxis()

    return ax

def plot_first_component(ax):

    path = Path('./results/full_run')
    
    tasks =   ['grasp'] 
    filters = ['beta']

    # load data outside of loop
    data = {f'{t}_{f}': get_data_first_components(path/t/f) for t, f in product(tasks, filters)}

    # Calculate samples covariance matrices
    move, _, rest, _ = get_covs_per_class(data)

    ax = plot_first_components_distributions(ax, move, rest)

    ax.tick_params(axis='both', which='major', labelsize='x-large')

    return ax

def make(path):
    npcs = 10
    filters = ('beta', 'hg', 'betahg')

    path_cross_task = Path(r'results/cross_task/')
    path_full =       Path(r'results/full_run/')
    
    decoding_data = [get_data(path_full/'imagine'/filter_, 'full') for filter_ in filters][BETA] 
    cross_data =    [get_data(path_cross_task/filter_,     'full') for filter_ in filters][BETA]
    decoding_data_exec = [get_data(path_full/'grasp'/filter_, 'full') for filter_ in filters][BETA] 
    # decoding_data = [get_data(path_full/'imagine'/filter_, 'full') for filter_ in filters][HG] 
    # cross_data =    [get_data(path_cross_task/filter_,     'full') for filter_ in filters][HG]
    # decoding_data = [get_data(path_full/'imagine'/filter_, 'full') for filter_ in filters][BHG] 
    # cross_data =    [get_data(path_cross_task/filter_,     'full') for filter_ in filters][BHG]
    
    transfer_learning = np.load(path/f'pc{npcs}/grasp/beta/transfer_auc.npy').reshape(8, 7)  # TODO
    # transfer_learning = np.load(path/f'pc{npcs}/grasp/betahg/transfer_auc.npy').reshape(8, 7)  # TODO

    grid = GridSpec(2, 5)
    fig = plt.figure(figsize=(16, 9))

    ax_source = plot_panel_source(fig.add_subplot(grid[0, 2:]), transfer_learning)
    
    ax_cross_task = plot_cross_task_panel(fig.add_subplot(grid[0, 0:2]), cross_data)
    ax_cross_task = add_to_cross_task_plot(ax_cross_task, decoding_data, cross_data)

    ax_first_component = plot_first_component(fig.add_subplot(grid[1, 0]))
    
    within_results = decoding_data_exec[2, :, :, 0, 1].mean(axis=1)
    ax_matrix = plot_matrix(fig.add_subplot(grid[1, 1:3]), transfer_learning, within_results)

    ax_cross_task.set_title('Cross-task', fontsize='xx-large')
    ax_cross_task.set_yticks(np.arange(10)/10)
    ax_cross_task.set_ylim(.4, .8)

    ax_matrix.set_title('Cross-participant', fontsize='xx-large')
    ax_cross_task.set_ylabel('Area under the curve', fontsize='xx-large')

    fig.tight_layout()

    # fig.savefig('tmp.png')
    # return
    fig.savefig('./figures/figure_4_transfer_learning.png')
    fig.savefig('./figures/figure_4_transfer_learning.svg')

