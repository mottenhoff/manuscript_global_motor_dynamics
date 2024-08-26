from pathlib import Path
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_1samp

from libs import mappings
from libs.figures.figure_4_cross_task import plot_panel as plot_panel_cross_task
from libs.figures.figure_4_variation_in_pc import load_covariance_matrices
from libs.figures.figure_4_variation_in_pc import plot_panel as plot_panel_first_components

BETA, HG, BHG = 0, 1, 2
AUC = 0
TEST = 1

PCS = [3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
N_PPTS = len(mappings.PPTS)
N_FOLDS = 10

PC_IDX = -3
PPT_IDX = -2
TEN_PCS = 10
TEN_PCS_IDX = 2

def fdrcorrection(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.array(p, dtype=np.float64)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def annote_significance_above_max_y_value(ax, y_values: list, p_values: list):

    is_sig =  lambda p_value: '***' if p_value < 0.001 else \
                              '**'  if p_value < 0.01  else \
                              '*'   if p_value < 0.05  else \
                              'n.s'
    
    x_ticks = ax.get_xticks()
    for x_tick, y_value, p_value in zip(x_ticks, y_values, p_values):
        ax.annotate(f'{is_sig(p_value)}', xy=(x_tick, y_value+0.05), ha='center', va='bottom', fontsize='x-large')

    return ax

def sort_by_key_and_participants(path):

    pc_keys = [f'pc{pc}' for pc in PCS]
    ppt_keys = mappings.PPTS
    indices = list(product(pc_keys, ppt_keys))

    parts = path.parts
    return indices.index((parts[PC_IDX], parts[PPT_IDX]))

def get_data(main_path, type_):
    '''
    Returns AUCS of TEST: shape = [n_pcs, n_ppts, n_folds, n_repeats]
    '''
    paths = main_path.rglob(f'{type_}.npy')
    paths = sorted(paths, key=sort_by_key_and_participants)
    
    data = np.array([np.load(path)[:, AUC, TEST] for path in paths])  # shape = [10 x 6 x 10]
    data = data.reshape(-1, N_PPTS, N_FOLDS)

    return data

def get_cross_data(main_path):

    result = {}
    for direction in ['grasp-to-imagine', 'imagine-to-grasp']:
        paths = sorted(main_path.rglob(f'{direction}/results_test.npy'), key=lambda p: p.parts[-3])
        result[direction] = np.stack([np.load(path) for path in paths])

    return result

def load_transfer_learning_data(path, task, filter_):
    ''' Data is returned in a 8 x 7 matrix, following the
    pattern below. A diagonal of zeroes is added to make
    an 8x8 matrix and align the rows and column to reflect
    the correct participants.

    sanity_check = np.array([[2,3,4,5,6,7,8],
                             [1,3,4,5,6,7,8],
                             [1,2,4,5,6,7,8],
                             [1,2,3,5,6,7,8],
                             [1,2,3,4,6,7,8],
                             [1,2,3,4,5,7,8],
                             [1,2,3,4,5,6,8],
                             [1,2,3,4,5,6,7]])
    sc = np.vstack([np.insert(row, i, 0) for i, row in enumerate(sanity_check)])
    '''

    path = path/f'pc{TEN_PCS}'/f'{task}'/f'{filter_}'/'transfer_auc.npy'
    data = np.load(path).reshape(8, 7)
    
    # insert a diagonal of zeroes to create an 8x8 matrix
    data = np.vstack([np.insert(row, i, 0) for i, row in enumerate(data)])
    
    return data

def plot_panel_transfer_learning(ax, data):

    # Remove the zero diagonal
    data = [row[np.where(row!=0)] for row in data]

    pvalues = ttest_1samp(data, .5, axis=1, alternative='two-sided').pvalue
    pvalues = fdrcorrection(pvalues)

    ppts = mappings.PPTS
    colors = [mappings.color_map()[ppt] for ppt in ppts]

    # Create a violinplot for each column of the matrix
    vp = ax.violinplot(data, showmeans=True, showmedians=False, showextrema=False)

    for i, body in enumerate(vp['bodies']):
        body.set_facecolor('grey')

    vp['cmeans'].set_color('black')    
    
    # Customize the plot
    ax.set_xticks(np.arange(N_PPTS)+1)
    ax.set_xticklabels(mappings.PPTS, fontsize='x-large')

    ax.set_title('Cross-participant', fontsize='xx-large')
    ax.set_xlabel('Source participant', fontsize='xx-large')

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
        ax.scatter(x, d, s=20, c=cs, alpha=1) # 0.75

    return ax

def plot_matrix(ax, data, data_within):

    # Add within ppt scores on the diagonal
    data[np.arange(N_PPTS), np.arange(N_PPTS)] = data_within

    ax.imshow(data, cmap='plasma', vmin=0.25, vmax=1)

    for s, t in product(np.arange(data.shape[0]), np.arange(data.shape[1])):
        value = data[t, s]
        txt = f'{value:0.2f}' if value != 0 else 'n/a'
        ax.annotate(txt, (s, t), ha='center', va='center', color='w')

    ax.set_xlabel('Target participant', fontsize='xx-large')
    ax.set_ylabel('Source participant', fontsize='xx-large')
    
    ppts = [ppt[1:] for ppt in mappings.PPTS]

    ax.set_xticks(np.arange(len(ppts)))
    ax.set_xticklabels(ppts, fontsize='large')
    ax.set_yticks(np.arange(len(ppts)))
    ax.set_yticklabels(ppts, fontsize='large')
    ax.invert_yaxis()

    ax.set_title('Cross-participant', fontsize='xx-large')

    return ax

def make(path):
    filters = ['beta'] #, 'hg', 'betahg')
    task = ('grasp')  # For transfer learning

    path_cross_task = Path(r'results/cross_task/')
    path_full =       Path(r'results/full_run/')
    
    decoding_exec =  [get_data(path_full/'grasp'/filter_,   'full') for filter_ in filters][0]
    decoding_imag =  [get_data(path_full/'imagine'/filter_, 'full') for filter_ in filters][0]
    decoding_cross = [get_cross_data(path_cross_task/filter_) for filter_ in filters][0]
    decoding_trans = [load_transfer_learning_data(path, task, filter_) for filter_ in filters][0]
    covariance_move, covariance_rest = [load_covariance_matrices(path_full/'grasp'/filter_)\
                                        for filter_ in filters][0]

    grid = GridSpec(2, 5)
    fig = plt.figure(figsize=(16, 9))

    
    plot_panel_cross_task(fig.add_subplot(grid[0, 0:2]),
                          decoding_cross,
                          decoding_imag,
                          decoding_exec)
    
    plot_panel_transfer_learning(fig.add_subplot(grid[0, 2:]), 
                                 decoding_trans)
    
    plot_panel_first_components(fig.add_subplot(grid[1, 0]), covariance_move, covariance_rest)

    plot_matrix(fig.add_subplot(grid[1, 1:3]), 
                decoding_trans,
                decoding_exec[TEN_PCS_IDX, :, :].mean(axis=-1))

    fig.tight_layout()

    fig.savefig('./figures/figure_4_transfer_learning.png')
    fig.savefig('./figures/figure_4_transfer_learning.svg')